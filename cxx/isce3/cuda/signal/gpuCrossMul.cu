#include "gpuCrossMul.h"

#include "gpuFilter.h"
#include "gpuLooks.h"
#include "gpuSignal.h"
#include <isce3/signal/Filter.h>
#include <isce3/signal/Signal.h>
#include <isce3/cuda/except/Error.h>

#include <climits>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define THRD_PER_BLOCK 1024 // Number of threads per block (should always %32==0)

template <typename T>
__global__ void form_interferogram_g(thrust::complex<T> *ifgram,
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
   computes coherence from 2 SLC rasters
output:
    coh:         coherence calculated from power of 2 SLCs and their interferogram
                 size: m x n
input:
    ref_power:   reference SLC power
                 size: m x n
    sec_power:   secondary SLC power
                 size: m x n
    igram:       interferogram created from ref and sec SLCs
                 size: m x n
    n_elements:  total number of elements. m x n
*/
template <typename T>
__global__ void calculate_coherence_g(T *coh,
        const T* __restrict__ ref_power,
        const T* __restrict__ sec_power,
        const thrust::complex<T>* __restrict__ igram,
        size_t n_elements)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // make sure index within ifgram size bounds
    if (i < n_elements) {
        coh[i] = thrust::abs(igram[i]) / sqrt(ref_power[i] * sec_power[i]);
    }
}


/*
    flattens interferogram with range offset, range spacing, and wavelength
output:
    ifg :                       flattened interferogram (flattening done in-place)
                                size: m x n
input:
    ifg :                       unflattened interferogram (flattening done in-place)
                                size: m x n
    rg_offset:                  range offset (pixels) from geo2rdr to be applied to interferogram
                                size: m x n
    rg2phase_conversion_factor: range spacing and wavelength to be applied to interferogram
                                4.0 * PI * range_spacing / wavelength
    n_elements:                 total number of elements. m x n
*/
template <class T>
__global__ void flatten_g(thrust::complex<T> *ifg,
        const double* __restrict__ rg_offset,
        double rg2phase_conversion_factor,
        size_t n_elements)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n_elements) {
        auto range_offset_phase = rg2phase_conversion_factor * rg_offset[i];
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
        _multiLookEnabled = true;
    else
        _multiLookEnabled = false;
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
        _multiLookEnabled = true;
    else
        _multiLookEnabled = false;
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
 * @param[in] linesPerBlock number of rows of the block of data
 * @param[out] shiftImpact frequency response (a linear phase) to a sub-pixel shift in time domain introduced by upsampling followed by downsampling
 */
void lookdownShiftImpact(size_t oversample,
        size_t nfft,
        size_t linesPerBlock,
        thrust::host_vector<std::complex<float>> & shiftImpact)
{
    // number of elements in oversampled line
    const size_t ncols = oversample * nfft;

    // range frequencies given nfft and oversampling factor
    std::valarray<double> rangeFrequencies(ncols);

    // sampling interval in range
    double dt = 1.0 / oversample;

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
    // a linear phase in frequency domain, we compute the impact in frequency domain.

    // the constant shift based on the oversampling factor
    const double shift = (1.0 - 1.0 / oversample) / 2.0;

    // compute the frequency response of the subpixel shift in range direction
    thrust::host_vector<std::complex<float>> shiftImpactLine(ncols);
    for (size_t col=0; col < ncols; ++col) {
        double phase = -1.0*shift*2.0*M_PI*rangeFrequencies[col];
        shiftImpactLine[col] = std::complex<float>(std::cos(phase),
                                                   std::sin(phase));
    }

    // The imapct is the same for each range line. Therefore copying the line for the block
    for (size_t line = 0; line < linesPerBlock; ++line) {
        thrust::copy_n(shiftImpactLine.begin(), ncols,
                       shiftImpact.begin() + line * ncols);
    }
}

void isce3::cuda::signal::gpuCrossmul::
crossmul(isce3::io::Raster& refSlcRaster,
        isce3::io::Raster& secSlcRaster,
        isce3::io::Raster& ifgRaster,
        isce3::io::Raster& coherenceRaster,
        isce3::io::Raster* rngOffsetRaster) const
{
    // set flatten flag based range offset raster ptr value
    bool flatten = rngOffsetRaster ? true : false;

    // setting local lines per block to avoid modifying class member
    size_t linesPerBlock = _linesPerBlock;

    if (linesPerBlock > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "linesPerBlock > INT_MAX");

    // check consistency of input/output raster shapes
    size_t nrows = refSlcRaster.length();
    size_t ncols = refSlcRaster.width();

    if (ncols > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "ncols > INT_MAX");

    if (ifgRaster.length() != coherenceRaster.length())
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "interferogram and coherence rasters length do not match");

    if (ifgRaster.width() != coherenceRaster.width())
        throw isce3::except::LengthError(ISCE_SRCINFO(),
                "interferogram and coherence rasters width do not match");

    const auto output_rows = ifgRaster.length();
    const auto output_cols = ifgRaster.width();
    if (_multiLookEnabled) {
        // Making sure that the number of rows in each block (linesPerBlock)
        // to be an integer multiple of the number of azimuth looks.
        linesPerBlock = (_linesPerBlock / _azimuthLooks) * _azimuthLooks;

        // checking only multilook interferogram shape is sufficient
        // interferogram and coherence shapes checked to match above
        if (output_rows != nrows / _azimuthLooks)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "multilooked interferogram/coherence raster lengths of unexpected size");

        if (output_cols != ncols / _rangeLooks)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "multilooked interferogram/coherence raster widths of unexpected size");
    } else {
        // checking only multilook interferogram shape is sufficient
        // interferogram and coherence shapes checked to match above
        if (output_rows != nrows)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "full resolution input/output raster lengths do not match");

        if (output_cols != ncols)
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                    "full resolution input/output raster widths do not match");
    }

    const size_t linesPerBlockMultiLooked = linesPerBlock/_azimuthLooks;
    const size_t ncolsMultiLooked = ncols/_rangeLooks;

    // number of blocks to process
    size_t nblocks = nrows / linesPerBlock;
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * linesPerBlock) != 0) {
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
    if (_oversampleFactor * nfft > INT_MAX)
        throw isce3::except::LengthError(ISCE_SRCINFO(), "_oversampleFactor * nfft > INT_MAX");

    // set upsampling FFT plans if upsampling
    if (_oversampleFactor) {
        signalNoUpsample.rangeFFT(nfft, linesPerBlock);
        signalUpsample.rangeFFT(nfft*_oversampleFactor, linesPerBlock);
    }

    // set not upsampled parameters
    auto n_slc = nfft*linesPerBlock;

    // storage for a block of reference SLC data
    std::valarray<std::complex<float>> refSlcOrig(n_slc);
    thrust::device_vector<thrust::complex<float>> d_refSlcOrig(n_slc);

    // storage for a block of secondary SLC data
    std::valarray<std::complex<float>> secSlcOrig(n_slc);
    thrust::device_vector<thrust::complex<float>> d_secSlcOrig(n_slc);

    // set upsampled parameters
    auto n_slcUpsampled = _oversampleFactor * nfft * linesPerBlock;

    // upsampled block of reference SLC
    std::valarray<std::complex<float>> refSlcUpsampled(n_slcUpsampled);
    thrust::device_vector<thrust::complex<float>> d_refSlcUpsampled;
    if (_oversampleFactor > 1)
        d_refSlcUpsampled.resize(n_slcUpsampled);

    // upsampled block of secondary SLC
    thrust::device_vector<thrust::complex<float>> d_secSlcUpsampled;
    if (_oversampleFactor > 1)
        d_secSlcUpsampled.resize(n_slcUpsampled);

    // calculate and copy to device shiftImpact frequency response (a linear phase)
    // to a sub-pixel shift in time domain introduced by upsampling followed by downsampling
    thrust::host_vector<std::complex<float>> h_shiftImpact(n_slcUpsampled);
    thrust::device_vector<thrust::complex<float>> d_shiftImpact(n_slcUpsampled);
    lookdownShiftImpact(_oversampleFactor,
            nfft,
            linesPerBlock,
            h_shiftImpact);
    d_shiftImpact = h_shiftImpact;

    // interferogram
    auto n_full_res = ncols * linesPerBlock;
    thrust::host_vector<std::complex<float>> h_ifgram(n_full_res,
                                                      std::complex<float>(0.0, 0.0));
    thrust::device_vector<thrust::complex<float>> d_ifgram(n_full_res);

    // range offset
    std::valarray<double> rngOffset(n_full_res);
    thrust::device_vector<double> d_rngOffset;
    // only resize if we're using...
    if (flatten) {
        d_rngOffset.resize(n_full_res);
    }

    // multilooked products container and parameters
    std::valarray<std::complex<float>> ifgram_mlook;
    thrust::device_vector<thrust::complex<float>> d_ifgram_mlook;
    std::valarray<float> coherence;

    auto n_mlook = linesPerBlockMultiLooked * ncolsMultiLooked;
    thrust::device_vector<float> d_ref_power;
    thrust::device_vector<float> d_sec_power;
    thrust::device_vector<float> d_coh;

    // use multilook flag to correctly size SLC, power, and coherence arrays
    if (_multiLookEnabled) {
        ifgram_mlook.resize(n_mlook);
        d_ifgram_mlook.resize(n_mlook);

        coherence.resize(n_mlook);

        d_ref_power.resize(n_mlook);
        d_sec_power.resize(n_mlook);
        d_coh.resize(n_mlook);
    } else {
        coherence.resize(n_full_res);
    }

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid_hi((refSlcOrig.size()*_oversampleFactor+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);
    dim3 grid_reg((refSlcOrig.size()+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);
    dim3 grid_lo((linesPerBlockMultiLooked*ncolsMultiLooked+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // loop over all blocks
    for (size_t i_block = 0; i_block < nblocks; ++i_block) {
        std::cout << "i_block: " << i_block+1 << " of " << nblocks << std::endl;
        // start row for this block
        const auto rowStart = i_block * linesPerBlock;

        //number of lines of data in this block. linesThisBlock<= linesPerBlock
        //Note that linesPerBlock is fixed number of lines
        //linesThisBlock might be less than or equal to linesPerBlock.
        //e.g. if nrows = 512, and linesPerBlock = 100, then
        //linesThisBlock for last block will be 12
        const auto linesThisBlock = std::min(nrows - rowStart, linesPerBlock);

        // fill the valarray with zero before getting the block of the data
        // this effectively zero-pads SLC arrays in range up-to length nfft
        refSlcOrig = 0;
        secSlcOrig = 0;
        refSlcUpsampled = 0;

        // get a block of reference and secondary SLC data
        // and a block of range offsets
        // This will change once we have the functionality to
        // get a block of data directly in to a slice
        // This zero-pads SLCs in range
        std::valarray<std::complex<float>> dataLine(ncols);
        for (size_t line = 0; line < linesThisBlock; ++line){
            refSlcRaster.getLine(dataLine, rowStart + line);
            refSlcOrig[std::slice(line*nfft, ncols, 1)] = dataLine;
            secSlcRaster.getLine(dataLine, rowStart + line);
            secSlcOrig[std::slice(line*nfft, ncols, 1)] = dataLine;
        }
        auto slc_size = n_slc * sizeof(thrust::complex<float>);
        checkCudaErrors(cudaMemcpy(d_refSlcOrig.data().get(), &refSlcOrig[0],
                    slc_size, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_secSlcOrig.data().get(), &secSlcOrig[0],
                    slc_size, cudaMemcpyHostToDevice));

        auto oversample_f = static_cast<float>(_oversampleFactor);
        if (_oversampleFactor > 1) {
            // upsample reference and secondary. done on device
            upsample(signalNoUpsample,
                    signalUpsample,
                    d_refSlcOrig.data().get(),
                    d_refSlcUpsampled.data().get(),
                    d_shiftImpact.data().get());
            upsample(signalNoUpsample,
                    signalUpsample,
                    d_secSlcOrig.data().get(),
                    d_secSlcUpsampled.data().get(),
                    d_shiftImpact.data().get());

            form_interferogram_g<<<grid_reg, block>>>(
                    d_ifgram.data().get(),
                    d_refSlcUpsampled.data().get(),
                    d_secSlcUpsampled.data().get(),
                    linesThisBlock, ncols, nfft, _oversampleFactor, oversample_f);
        } else {
            form_interferogram_g<<<grid_reg, block>>>(
                    d_ifgram.data().get(),
                    d_refSlcOrig.data().get(),
                    d_secSlcOrig.data().get(),
                    linesThisBlock, ncols, nfft, _oversampleFactor, oversample_f);
        }

        // Check for any kernel errors
        checkCudaErrors(cudaPeekAtLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // flatten post interferogram - ala CrossMultipy + Flatten
        if (flatten) {
            // Read range offsets
            std::valarray<double> offsetLine(ncols);
            for (size_t line = 0; line < linesThisBlock; ++line){
                rngOffsetRaster->getLine(offsetLine, rowStart + line);
                rngOffset[std::slice(line*ncols, ncols, 1)] = offsetLine + _offsetStartingRangeShift / _rangePixelSpacing;
            }
            checkCudaErrors(cudaMemcpy(d_rngOffset.data().get(), &rngOffset[0],
                        n_full_res*sizeof(double), cudaMemcpyHostToDevice));

            double rg2phase_conversion_factor = 4.0*M_PI*_rangePixelSpacing/_wavelength;
            flatten_g<<<grid_reg, block>>>(d_ifgram.data().get(),
                    d_rngOffset.data().get(),
                    rg2phase_conversion_factor,
                    linesThisBlock*ncols);

            // Check for any kernel errors
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

        // compute ifg according to multilook flag
        if (_multiLookEnabled) {
            // compute multilooked interferogram
            // reduce ncols*nrow to ncolsMultiLooked*linesPerBlockMultiLooked
            multilooks_g<<<grid_lo, block>>>(
                    d_ifgram_mlook.data().get(),
                    d_ifgram.data().get(),
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
            checkCudaErrors(cudaMemcpy(&ifgram_mlook[0],
                        d_ifgram_mlook.data().get(),
                        n_mlook*sizeof(thrust::complex<float>),
                        cudaMemcpyDeviceToHost));

            ifgRaster.setBlock(ifgram_mlook, 0, rowStart/_azimuthLooks,
                        ncols/_rangeLooks, linesThisBlock/_azimuthLooks);

            // compute mulitlooked coherence
            // set grid size based on multilook flag
            auto grid = grid_lo;

            // calculate power of reference SLC
            thrust::complex<float>* ref_slc =
                    _oversampleFactor > 1 ? d_refSlcUpsampled.data().get()
                                    : d_refSlcOrig.data().get();
            multilooks_power_g<<<grid, block>>>(
                    d_ref_power.data().get(),
                    ref_slc,
                    2,
                    _oversampleFactor*nfft,         // n columns hi res
                    ncolsMultiLooked,               // n columns lo res
                    _azimuthLooks,                  // row resize factor of hi to lo
                    _oversampleFactor*_rangeLooks,  // col resize factor of hi to lo
                    n_mlook,                        // number of lo res elements
                    float(_oversampleFactor*_azimuthLooks*_rangeLooks));

            // Check for any kernel errors
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // calculate power of secondary SLC
            thrust::complex<float> *sec_slc = _oversampleFactor > 1 ?
                        d_secSlcUpsampled.data().get() :
                        d_secSlcOrig.data().get();
            multilooks_power_g<<<grid_lo, block>>>(
                    d_sec_power.data().get(),
                    sec_slc,
                    2,
                    _oversampleFactor*nfft,
                    ncolsMultiLooked,
                    _azimuthLooks,                  // row resize factor of hi to lo
                    _oversampleFactor*_rangeLooks,  // col resize factor of hi to lo
                    n_mlook,                        // number of lo res elements
                    float(_oversampleFactor*_azimuthLooks*_rangeLooks));

            // Check for any kernel errors
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // perform coherence calculation
            thrust::complex<float> *ifg = d_ifgram_mlook.data().get();
            const size_t ifg_sz = ifgram_mlook.size();
            calculate_coherence_g<<<grid, block>>>(d_coh.data().get(),
                    d_ref_power.data().get(),
                    d_sec_power.data().get(),
                    ifg, ifg_sz);

            // Check for any kernel errors
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            // copy coherence data to HOST
            const size_t n_coh = n_mlook;
            checkCudaErrors(cudaMemcpy(&coherence[0], d_coh.data().get(),
                        n_coh * sizeof(float), cudaMemcpyDeviceToHost));

            // set blocks accordingly
            coherenceRaster.setBlock(coherence, 0, rowStart/_azimuthLooks,
                        ncols/_rangeLooks, linesThisBlock/_azimuthLooks);
        } else {
            // get data to HOST
            h_ifgram = d_ifgram;

            // set the block of interferogram
            ifgRaster.setBlock(h_ifgram.data(), 0, rowStart, ncols,
                                   linesThisBlock);

            // fill coherence with ones (no need to compute result)
            coherence = 1.0;

            // set the block of coherence
            coherenceRaster.setBlock(coherence, 0, rowStart, ncols, linesThisBlock);
        }
    }
}
