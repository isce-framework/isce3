#include "gpuCrossMul.h"

#include "gpuFilter.h"
#include "gpuLooks.h"
#include "gpuSignal.h"
#include <isce3/signal/Filter.h>
#include <isce3/signal/Signal.h>
#include <isce3/cuda/except/Error.h>

#include <climits>

#define THRD_PER_BLOCK 1024 // Number of threads per block (should always %32==0)

/*
output
    thrust::complex *ifgram (n_cols*n_rows)
input
    thrust::complex *refSlcUp ((oversample*n_fft)*n_rows) nfft >= where ncols
    thrust::complex *secSlcUp
    size_t n_rows
    size_t n_cols
    size_t n_fft
    int oversample_int
    float oversample_float
*/
template <typename T>
__global__ void interferogram_g(thrust::complex<T> *ifgram,
        const thrust::complex<T>* __restrict__ refSlcUp,
        const thrust::complex<T>* __restrict__ secSlcUp,
        size_t n_rows,
        size_t n_cols,
        size_t n_fft,
        int oversample_int,
        T oversample_float)
{
    // get 1-d interferogram index
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // make sure index within ifgram size bounds
    if (i < n_rows * n_cols) {
        // break up 1-d index into 2-d index
        auto i_row = i / n_cols;
        auto i_col = i % n_cols;

        // local accumulation variable
        auto accumulation = thrust::complex<T>(0.0, 0.0);

        // accumulate crossmultiplied oversampled pixels
        // oversample_int > 0 so crossmultiply will always be calculated
        for (int j = 0; j < oversample_int; ++j) {
            // get 1-d, maybe oversampled, index based on 2-d index
            // i_row * n_fft + i_col = 1-d index w/o oversampling
            // oversample_int * (..) = first 1-d index w/ oversampling
            // (...) + j = j-th oversampled index
            auto i_up = oversample_int * (i_row * n_fft + i_col) + j;

            // get values from SLC rasters and crossmultiply
            auto ref_val = refSlcUp[i_up];
            auto sec_val_conj = thrust::conj(secSlcUp[i_up]);
            accumulation += ref_val * sec_val_conj;
        }
        // normalize by oversample factor
        ifgram[i] = accumulation / oversample_float;
    }
}


/*
   computes coherence from 2 SLC amplitude rasters
output:
    ref_amp_to_coh: initially input SLC amplitude that later overwritten with coherence
                    size: m x n
input:
    ref_amp_to_coh: initially input SLC amplitude that later overwritten with coherence
                    size: m x n
    sec_amp:        another input SLC amplitude
                    size: m x n
    igram:          interferogram created from ref and sec SLCs
                    size: m x n
    n_elements:     total number of elements. m x n
*/
template <typename T>
__global__ void calculate_coherence_g(T *ref_amp_to_coh,
        const T* __restrict__ sec_amp,
        const thrust::complex<T>* __restrict__ igram,
        size_t n_elements)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // make sure index within ifgram size bounds
    // overwrite existing amplitude value as it's never used again
    if (i < n_elements) {
        ref_amp_to_coh[i] = thrust::abs(igram[i]) / sqrt(ref_amp_to_coh[i] * sec_amp[i]);
    }
}


template <class T>
__global__ void flatten(thrust::complex<T> *ifg,
        const double* __restrict__ rg_offset,
        double offset_phase,
        size_t n_elements)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n_elements) {
        // offset_phase = 4.0 * M_PI * range_spacing / wavelength
        auto range_offset_phase = offset_phase * rg_offset[i];
        thrust::complex<T> shift(std::cos(range_offset_phase), -std::sin(range_offset_phase));
        ifg[i] *= shift;
    }
}


/** Set number of range looks */
void isce3::cuda::signal::gpuCrossmul::
rangeLooks(int rngLks) {
    if (rngLks < 1) {
        std::string error_msg = "ERROR CUDA crossmul range multilook < 1";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    _rangeLooks = rngLks;
    if (_azimuthLooks > 1 or _rangeLooks > 1)
        _doMultiLook = true;
    else
        _doMultiLook = false;
}

/** Set number of azimuth looks */
void isce3::cuda::signal::gpuCrossmul::
azimuthLooks(int azLks) {
    if (azLks < 1) {
        std::string error_msg = "ERROR CUDA crossmul azimuth multilook < 1";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_msg);
    }

    _azimuthLooks = azLks;
    if (_azimuthLooks > 1 or _rangeLooks > 1)
        _doMultiLook = true;
    else
        _doMultiLook = false;
}

void isce3::cuda::signal::gpuCrossmul::
doppler(isce3::core::LUT1d<double> refDoppler,
        isce3::core::LUT1d<double> secDoppler)
{
    _refDoppler = refDoppler;
    _secDoppler = secDoppler;
}


/**
 * @param[in] oversample upsampling factor
 * @param[in] nfft fft length in range direction
 * @param[in] blockRows number of rows of the block of data
 * @param[out] shiftImpact frequency responce (a linear phase) to a sub-pixel shift in time domain introduced by upsampling followed by downsampling
 */
void lookdownShiftImpact(size_t oversample,
        size_t nfft,
        size_t blockRows,
        std::valarray<std::complex<float>> &shiftImpact)
{
    // range frequencies given nfft and oversampling factor
    std::valarray<double> rangeFrequencies(oversample*nfft);

    // sampling interval in range
    double dt = 1.0/oversample;

    // get the vector of range frequencies
    isce3::signal::fftfreq(dt, rangeFrequencies);


    // in the process of upsampling the SLCs, creating upsampled interferogram
    // and then looking down the upsampled interferogram to the original size of
    // the SLCs, a shift is introduced in range direction.
    // As an example for a signal with length of 5 and :
    // original sample locations:   0       1       2       3        4
    // upsampled sample locations:  0   0.5 1  1.5  2  2.5  3   3.5  4   4.5
    // Looked dow sample locations:   0.25    1.25    2.25    3.25    4.25
    // Obviously the signal after looking down would be shifted by 0.25 pixel in
    // range comared to the original signal. Since a shift in time domain introduces
    // a liner phase in frequency domain, we compute the impact in frequency domain.

    // the constant shift based on the oversampling factor
    double shift = (1.0 - 1.0/oversample)/2.0;

    // compute the frequency response of the subpixel shift in range direction
    std::valarray<std::complex<float>> shiftImpactLine(oversample*nfft);
    for (size_t col=0; col<shiftImpactLine.size(); ++col){
        double phase = -1.0*shift*2.0*M_PI*rangeFrequencies[col];
        shiftImpactLine[col] = std::complex<float> (std::cos(phase),
                                                    std::sin(phase));
    }

    // The imapct is the same for each range line. Therefore copying the line for the block
    for (size_t line = 0; line < blockRows; ++line){
            shiftImpact[std::slice(line*nfft*oversample, nfft*oversample, 1)] = shiftImpactLine;
    }
}


void isce3::cuda::signal::gpuCrossmul::
crossmul(isce3::io::Raster& referenceSLC,
        isce3::io::Raster& secondarySLC,
        isce3::io::Raster& interferogram)
{
    _doCommonRangeBandFilter = false;
    isce3::io::Raster rngOffsetRaster("/vsimem/dummy", 1, 1, 1, GDT_CFloat32, "ENVI");
    isce3::io::Raster coherence("/vsimem/dummyCoh", 1, 1, 1, GDT_Float32, "ENVI");
    crossmul(referenceSLC,
            secondarySLC,
            rngOffsetRaster,
            interferogram,
            coherence);

}

void isce3::cuda::signal::gpuCrossmul::
crossmul(isce3::io::Raster& referenceSLC,
        isce3::io::Raster& secondarySLC,
        isce3::io::Raster& interferogram,
        isce3::io::Raster& coherence)
{
    _doCommonRangeBandFilter = false;
    isce3::io::Raster rngOffsetRaster("/vsimem/dummy", 1, 1, 1, GDT_CFloat32, "ENVI");
    crossmul(referenceSLC,
            secondarySLC,
            rngOffsetRaster,
            interferogram,
            coherence);

}

void isce3::cuda::signal::gpuCrossmul::
crossmul(isce3::io::Raster& referenceSLC,
        isce3::io::Raster& secondarySLC,
        isce3::io::Raster& rngOffsetRaster,
        isce3::io::Raster& interferogram,
        isce3::io::Raster& coherenceRaster) const
{
    size_t nrows = referenceSLC.length();
    size_t ncols = referenceSLC.width();

    if (ncols > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "ncols > INT_MAX");

    // setting the parameters of the multi-looking oject
    size_t rowsPerBlock = _rowsPerBlock;
    if (_doMultiLook) {
        // Making sure that the number of rows in each block (rowsPerBlock)
        // to be an integer number of azimuth looks.
        rowsPerBlock = (_rowsPerBlock/_azimuthLooks)*_azimuthLooks;
    }

    if (rowsPerBlock > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "rowsPerBlock > INT_MAX");

    size_t blockRowsMultiLooked = rowsPerBlock/_azimuthLooks;
    size_t ncolsMultiLooked = ncols/_rangeLooks;

    // number of blocks to process
    size_t nblocks = nrows / rowsPerBlock;
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * rowsPerBlock) != 0) {
        nblocks += 1;
    }

    // signal object for upsampling
    isce3::cuda::signal::gpuSignal<float> signalNoUpsample(CUFFT_C2C);
    isce3::cuda::signal::gpuSignal<float> signalUpsample(CUFFT_C2C);

    // Compute FFT size (power of 2)
    size_t nfft;
    signalNoUpsample.nextPowerOfTwo(ncols, nfft);

    if (nfft > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "nfft > INT_MAX");
    if (_oversample * nfft > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "_oversample * nfft > INT_MAX");

    // set upsampling FFT plans
    signalNoUpsample.rangeFFT(nfft, rowsPerBlock);
    signalUpsample.rangeFFT(nfft*_oversample, rowsPerBlock);

    // set not upsampled parameters
    auto n_slc = nfft*rowsPerBlock;
    auto slc_size = n_slc * sizeof(thrust::complex<float>);

    // storage for a block of reference SLC data
    std::valarray<std::complex<float>> refSlcOrig(n_slc);
    thrust::complex<float> *d_refSlcOrig;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_refSlcOrig), slc_size));

    // storage for a block of secondary SLC data
    std::valarray<std::complex<float>> secSlcOrig(n_slc);
    thrust::complex<float> *d_secSlcOrig;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_secSlcOrig), slc_size));

    // set upsampled parameters
    auto n_slcUpsampled = _oversample * nfft * rowsPerBlock;
    auto slcUpsampled_size = n_slcUpsampled * sizeof(thrust::complex<float>);

    // upsampled block of reference SLC
    std::valarray<std::complex<float>> refSlcUpsampled(n_slcUpsampled);
    thrust::complex<float> *d_refSlcUpsampled;
    if (_oversample > 1)
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_refSlcUpsampled), slcUpsampled_size));

    // upsampled block of secondary SLC
    thrust::complex<float> *d_secSlcUpsampled;
    if (_oversample > 1)
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_secSlcUpsampled), slcUpsampled_size));

    // calculate and copy to device shiftImpact frequency responce (a linear phase)
    // to a sub-pixel shift in time domain introduced by upsampling followed by downsampling
    std::valarray<std::complex<float>> shiftImpact(n_slcUpsampled);
    thrust::complex<float> *d_shiftImpact;
    lookdownShiftImpact(_oversample,
            nfft,
            rowsPerBlock,
            shiftImpact);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_shiftImpact), slcUpsampled_size));
    checkCudaErrors(cudaMemcpy(d_shiftImpact, &shiftImpact[0], slcUpsampled_size, cudaMemcpyHostToDevice));

    // interferogram
    auto n_ifgram = ncols * rowsPerBlock;
    auto ifgram_size = n_ifgram * sizeof(thrust::complex<float>);
    std::valarray<std::complex<float>> ifgram(n_ifgram);
    thrust::complex<float> *d_ifgram;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_ifgram), ifgram_size));

    // range offset
    std::valarray<double> rngOffset(n_ifgram);
    double *d_rngOffset;
    auto rngOffset_size = n_ifgram*sizeof(double);
    if (_doCommonRangeBandFilter) {
        // only malloc if we're using...
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_rngOffset), rngOffset_size));
    }

    // multilooked products container and parameters
    std::valarray<std::complex<float>> ifgram_mlook(0);
    std::valarray<float> coherence(0);
    auto n_mlook = blockRowsMultiLooked * ncolsMultiLooked;
    auto mlook_size = n_mlook*sizeof(float);

    // CUDA device memory allocation
    thrust::complex<float> *d_ifgram_mlook;
    float *d_ref_amp_mlook;
    float *d_sec_amp_mlook;

    if (_doMultiLook) {
        ifgram_mlook.resize(n_mlook);
        coherence.resize(n_mlook);
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_ifgram_mlook), 2*mlook_size)); // 2* because imaginary
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_ref_amp_mlook), mlook_size));
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_sec_amp_mlook), mlook_size));
    }

    // filter objects
    isce3::cuda::signal::gpuAzimuthFilter<float> azimuthFilter;

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid_hi((refSlcOrig.size()*_oversample+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);
    dim3 grid_reg((refSlcOrig.size()+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);
    dim3 grid_lo((blockRowsMultiLooked*ncolsMultiLooked+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // configure azimuth filter
    if (_doCommonAzimuthBandFilter) {
        azimuthFilter.constructAzimuthCommonbandFilter(
                _refDoppler,
                _secDoppler,
                _commonAzimuthBandwidth,
                _prf,
                _beta,
                nfft, 
                rowsPerBlock);
    }

    // loop over all blocks
    for (size_t i_block = 0; i_block < nblocks; ++i_block) {
        std::cout << "i_block: " << i_block+1 << " of " << nblocks << std::endl;
        // start row for this block
        size_t rowStart;
        rowStart = i_block * rowsPerBlock;

        //number of lines of data in this block. rowsThisBlock<= rowsPerBlock
        //Note that rowsPerBlock is fixed number of lines
        //rowsThisBlock might be less than or equal to rowsPerBlock.
        //e.g. if nrows = 512, and rowsPerBlock = 100, then
        //rowsThisBlock for last block will be 12
        size_t rowsThisBlock;
        if ((rowStart + rowsPerBlock) > nrows) {
            rowsThisBlock = nrows - rowStart;
        } else {
            rowsThisBlock = rowsPerBlock;
        }

        // fill the valarray with zero before getting the block of the data
        refSlcOrig = 0;
        secSlcOrig = 0;
        refSlcUpsampled = 0;
        ifgram = 0;

        // get a block of reference and secondary SLC data
        // and a block of range offsets
        // This will change once we have the functionality to
        // get a block of data directly in to a slice
        std::valarray<std::complex<float>> dataLine(ncols);
        for (size_t line = 0; line < rowsThisBlock; ++line){
            referenceSLC.getLine(dataLine, rowStart + line);
            refSlcOrig[std::slice(line*nfft, ncols, 1)] = dataLine;
            secondarySLC.getLine(dataLine, rowStart + line);
            secSlcOrig[std::slice(line*nfft, ncols, 1)] = dataLine;
        }
        checkCudaErrors(cudaMemcpy(d_refSlcOrig, &refSlcOrig[0], slc_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_secSlcOrig, &secSlcOrig[0], slc_size, cudaMemcpyHostToDevice));

        // apply azimuth filter (do inplace)
        if (_doCommonAzimuthBandFilter) {
            azimuthFilter.filter(d_refSlcOrig);
            azimuthFilter.filter(d_secSlcOrig);
        }

        auto oversample_f = static_cast<float>(_oversample);
        if (_oversample > 1) {
            // upsample reference and secondary. done on device
            upsample(signalNoUpsample,
                    signalUpsample,
                    d_refSlcOrig,
                    d_refSlcUpsampled,
                    d_shiftImpact);
            upsample(signalNoUpsample,
                    signalUpsample,
                    d_secSlcOrig,
                    d_secSlcUpsampled,
                    d_shiftImpact);

            interferogram_g<<<grid_reg, block>>>(
                    d_ifgram,
                    d_refSlcUpsampled,
                    d_secSlcUpsampled,
                    rowsThisBlock, ncols, nfft, _oversample, oversample_f);
        } else {
            interferogram_g<<<grid_reg, block>>>(
                    d_ifgram,
                    d_refSlcOrig,
                    d_secSlcOrig,
                    rowsThisBlock, ncols, nfft, _oversample, oversample_f);
        }

        // Check for any kernel errors
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // flatten post interferogram - ala CrossMultipy + Flatten
        if (_doCommonRangeBandFilter) {
            // Read range offsets
            std::valarray<double> offsetLine(ncols);
            for (size_t line = 0; line < rowsThisBlock; ++line){
                rngOffsetRaster.getLine(offsetLine, rowStart + line);
                rngOffset[std::slice(line*ncols, ncols, 1)] = offsetLine;
            }
            checkCudaErrors(cudaMemcpy(d_rngOffset, &rngOffset[0], rngOffset_size, cudaMemcpyHostToDevice));

            double offset_phase = 4.0*M_PI*_rangePixelSpacing/_wavelength;
            flatten<<<grid_reg, block>>>(d_ifgram,
                    d_rngOffset,
                    offset_phase,
                    rowsThisBlock*ncols);
        }

        if (_doMultiLook) {

            // reduce ncols*nrow to ncolsMultiLooked*blockRowsMultiLooked
            multilooks_g<<<grid_lo, block>>>(
                    d_ifgram_mlook,
                    d_ifgram,
                    ncols,                          // n columns hi res
                    ncolsMultiLooked,               // n cols lo res
                    _azimuthLooks,                  // row resize factor of hi to lo
                    _rangeLooks,                    // col resize factor of hi to lo
                    n_mlook,                        // number of lo res elements
                    float(_azimuthLooks*_rangeLooks));

            // Check for any kernel errors
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // get data to HOST
            checkCudaErrors(cudaMemcpy(&ifgram_mlook[0], d_ifgram_mlook, mlook_size*2, cudaMemcpyDeviceToHost));

            interferogram.setBlock(ifgram_mlook, 0, rowStart/_azimuthLooks,
                        ncols/_rangeLooks, rowsThisBlock/_azimuthLooks);

            if (_oversample > 1) {
                // write reduce+abs and set blocks
                multilooks_power_g<<<grid_lo, block>>>(
                        d_ref_amp_mlook,
                        d_refSlcUpsampled,
                        2,
                        _oversample*nfft,               // n columns hi res
                        ncolsMultiLooked,               // n columns lo res
                        _azimuthLooks,                  // row resize factor of hi to lo
                        _oversample*_rangeLooks,        // col resize factor of hi to lo
                        n_mlook,                        // number of lo res elements
                        float(_oversample*_azimuthLooks*_rangeLooks));
            } else {
                multilooks_power_g<<<grid_lo, block>>>(
                        d_ref_amp_mlook,
                        d_refSlcOrig,
                        2,
                        _oversample*nfft,               // n columns hi res
                        ncolsMultiLooked,               // n columns lo res
                        _azimuthLooks,                  // row resize factor of hi to lo
                        _oversample*_rangeLooks,        // col resize factor of hi to lo
                        n_mlook,                        // number of lo res elements
                        float(_oversample*_azimuthLooks*_rangeLooks));
            }

            // Check for any kernel errors
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            if (_oversample > 1) {
                multilooks_power_g<<<grid_lo, block>>>(
                        d_sec_amp_mlook,
                        d_secSlcUpsampled,
                        2,
                        _oversample*nfft,
                        ncolsMultiLooked,
                        _azimuthLooks,                  // row resize factor of hi to lo
                        _oversample*_rangeLooks,        // col resize factor of hi to lo
                        n_mlook,                        // number of lo res elements
                        float(_oversample*_azimuthLooks*_rangeLooks));
            } else {
                multilooks_power_g<<<grid_lo, block>>>(
                        d_sec_amp_mlook,
                        d_secSlcOrig,
                        2,
                        _oversample*nfft,
                        ncolsMultiLooked,
                        _azimuthLooks,                  // row resize factor of hi to lo
                        _oversample*_rangeLooks,        // col resize factor of hi to lo
                        n_mlook,                        // number of lo res elements
                        float(_oversample*_azimuthLooks*_rangeLooks));
            }

            // Check for any kernel errors
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // perform coherence calculation in place overwriting d_ifgram_mlook
            calculate_coherence_g<<<grid_lo, block>>>(d_ref_amp_mlook,
                    d_sec_amp_mlook,
                    d_ifgram_mlook,
                    ifgram_mlook.size());

            // Check for any kernel errors
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // get data to HOST from overwritten multilooked reference amplitude
            checkCudaErrors(cudaMemcpy(&coherence[0], d_ref_amp_mlook, mlook_size, cudaMemcpyDeviceToHost));

            // set blocks accordingly
            coherenceRaster.setBlock(coherence, 0, rowStart/_azimuthLooks,
                        ncols/_rangeLooks, rowsThisBlock/_azimuthLooks);

        } else {
            // get data to HOST
            checkCudaErrors(cudaMemcpy(&ifgram[0], d_ifgram, ifgram_size, cudaMemcpyDeviceToHost));

            // set the block of interferogram
            interferogram.setBlock(ifgram, 0, rowStart, ncols, rowsThisBlock);
        }

    }

    // liberate all device memory
    checkCudaErrors(cudaFree(d_refSlcOrig));
    checkCudaErrors(cudaFree(d_secSlcOrig));
    if (_oversample > 1) {
        checkCudaErrors(cudaFree(d_refSlcUpsampled));
        checkCudaErrors(cudaFree(d_secSlcUpsampled));
    }
    checkCudaErrors(cudaFree(d_shiftImpact));
    checkCudaErrors(cudaFree(d_ifgram));
    if (_doCommonRangeBandFilter) {
        checkCudaErrors(cudaFree(d_rngOffset));
    }
    if (_doMultiLook) {
        checkCudaErrors(cudaFree(d_ifgram_mlook));
        checkCudaErrors(cudaFree(d_ref_amp_mlook));
        checkCudaErrors(cudaFree(d_sec_amp_mlook));
    }

}
