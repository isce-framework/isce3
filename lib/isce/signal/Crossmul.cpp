#include "Crossmul.h"

isce::signal::Crossmul::
Crossmul(const isce::product::Product& referenceSLC,
         const isce::product::Product& secondarySLC,
         int numberOfRangeLooks,
         int numberOfAzimuthLooks,
         std::string outputInterferogramFile)
{
    // Get sizes
    nrows = referenceSLC.length();
    ncols = referenceSLC.width();
    // Check consistency
    assert(nrows == int(secondarySLC.length()));
    assert(ncols == int(secondarySLC.width()));

    rowsLooks = numberOfRangeLooks;
    colsLooks = numberOfAzimuthLooks;

    // Compute sizes after looks
    nrows_ifgram = nrows / numberOfAzimuthLooks;
    ncols_ifgram = ncols / numberOfRangeLooks;
    
    
    // Open raster for writing
    isce::io::Raster ifgramRaster(outputInterferogramFile, ncols_ifgram, nrows_ifgram, 1,
                                          GDT_CFloat32, "ISCE");

      	
}

isce::signal::Crossmul::
crossmul(isce::io::Raster& referenceSLC,
        isce::io::Raster& secondarySLC,
        int numberOfRangeLooks,
        int numberOfAzimuthLooks,
        double commonAzimuthBandwidth,
        isce::io::Raster& interferogram)
{
    // Compute FFT size (power of 2)
    nfft = ncols; //nextPow2(ncols);
     
    // it should be determined somehow
    int blockRows = 100;
    int oversample = 2;
    // storage for blocks of data (reference and secondary slc)
    std::valarray<std::complex<float>> refSlc(ncols*blockRows*rowsLooks);
    std::valarray<std::complex<float>> secSlc(ncols*blockRows*rowsLooks);
    std::valarray<std::complex<float>> refSpectrum(nfft*blockRows*rowsLooks);
    std::valarray<std::complex<float>> secSpectrum(nfft*blockRows*rowsLooks);

    // upsampled reference and secondary spectrum and signal
    std::valarray<std::complex<float>> refSpectrumUpsampled(oversample*nfft*blockRows*rowsLooks);
    std::valarray<std::complex<float>> secSpectrumUpsampled(oversample*nfft*blockRows*rowsLooks);
    std::valarray<std::complex<float>> refSlcUpsampled(oversample*nfft*blockRows*rowsLooks);
    std::valarray<std::complex<float>> secSlcUpsampled(oversample*nfft*blockRows*rowsLooks);

    // interferogram at different stages
    std::valarray<std::complex<float>> ifgramUpsampled(oversample*ncols*blockRows*rowsLooks);

    //a signal object for refSlc
    isce::signal::Signal refSignal;

    //a signal object for secSlc
    isce::signal::Signal secSignal;

    // make fft plans for the reference SLC 
    refSignal.forwardRangeFFT(refSlc, refSpectrum, ncols, blockRows*rowsLooks, nfft, blockLength);
    refSignal.inverseRangeFFT(refSpectrumUpsampled, refSlcUpsampled, nfft*oversample, blockRows*rowsLooks, nfft*oversample, blockRows*rowsLooks);

    // make fft plans for the secondary SLC
    secSignal.forwardRangeFFT(secSlc, secSpectrum, ncols, blockRows*rowsLooks, nfft, blockLength);
    secSignal.inverseRangeFFT(secSpectrumUpsampled, secSlcUpsampled, nfft*oversample, blockRows*rowsLooks, nfft*oversample, blockRows*rowsLooks);

    // loop over all blocks
    for (size_t block = 0; block < nBlocks; ++block) {
        
        size_t rowStart;
        rowStart = block * blockRows*rowsLooks;

        // get a block of reference and secondary slc
        referenceSLC.getBlock(refSlc, 0, rowStart, ncols, blockRows*rowsLooks);
        secondarySLC.getBlock(secSlc, 0, rowStart, ncols, blockRows*rowsLooks);
    
        
        //commaon azimuth band-pass filter 
        
        // upsample the refernce and secondary SLCs
        refSignal.upsample(refSlc, refSlcUpsampled, blockRows*rowsLooks, nfft, oversample);
        secSignal.upsample(secSlc, secSlcUpsampled, blockRows*rowsLooks, nfft, oversample);
        
        // Compute oversampled interferogram data
        for (size_t i=0; i<oversample*ncols; ++i){
            ifgramUpsampled[i] = refSlcUpsampled[i]*std::conj(secSlcUpsampled[i]);
        }

        // Reclaim the extra oversample looks across
        for (size_t line=0; line<blockRows*rowsLooks; ++line){
            for (size_t i=0; i<ncols; ++i){
                std::complex<std::float> sum =(0,0);
                for (size_t j=0; j<oversample; ++j)
                    sum += ifgramUpsampled[line*(nfft*oversample) + 2*i + j];
                }
                ifgram[line*ncols + i] = sum;            
            }
        }

	// Take looks down (summing columns)

        // Take looks across (summing row blocks)
        
	// write data to file

    }
}


