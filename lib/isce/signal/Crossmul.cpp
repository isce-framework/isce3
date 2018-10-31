#include "Crossmul.h"

/*isce::signal::Crossmul::
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
*/

void isce::signal::Crossmul::
crossmul(isce::io::Raster& referenceSLC,
        isce::io::Raster& secondarySLC,
        int numberOfRangeLooks,
        int numberOfAzimuthLooks,
        double commonAzimuthBandwidth,
        isce::io::Raster& interferogram)
{

    nrows = referenceSLC.length();
    ncols = referenceSLC.width();

    assert(nrows == int(secondarySLC.length()));
    assert(ncols == int(secondarySLC.width()));

    // Compute FFT size (power of 2)
    nfft = ncols; //nextPow2(ncols);
    
    // it should be determined somehow
    int blockRows = 100;
    int oversample = 2;
    int nBlocks = 5;
    

    // storage for blocks of data (reference and secondary slc)
    std::valarray<std::complex<float>> refSlc(ncols*blockRows);
    std::valarray<std::complex<float>> secSlc(ncols*blockRows);
    std::valarray<std::complex<float>> refSpectrum(nfft*blockRows);
    std::valarray<std::complex<float>> secSpectrum(nfft*blockRows);

    // upsampled reference and secondary spectrum and signal
    std::valarray<std::complex<float>> refSpectrumUpsampled(oversample*nfft*blockRows);
    std::valarray<std::complex<float>> secSpectrumUpsampled(oversample*nfft*blockRows);
    std::valarray<std::complex<float>> refSlcUpsampled(oversample*nfft*blockRows);
    std::valarray<std::complex<float>> secSlcUpsampled(oversample*nfft*blockRows);

    // interferogram at different stages
    std::valarray<std::complex<float>> ifgramUpsampled(oversample*ncols*blockRows);
    std::valarray<std::complex<float>> ifgram(ncols*blockRows);

    //a signal object for refSlc
    isce::signal::Signal<float> refSignal;

    //a signal object for secSlc
    isce::signal::Signal<float> secSignal;

    // make fft plans for the reference SLC 
    refSignal.forwardRangeFFT(refSlc, refSpectrum, ncols, blockRows, nfft, blockRows);
    refSignal.inverseRangeFFT(refSpectrumUpsampled, refSlcUpsampled, nfft*oversample, blockRows, nfft*oversample, blockRows);

    // make fft plans for the secondary SLC
    secSignal.forwardRangeFFT(secSlc, secSpectrum, ncols, blockRows, nfft, blockRows);
    secSignal.inverseRangeFFT(secSpectrumUpsampled, secSlcUpsampled, nfft*oversample, blockRows, nfft*oversample, blockRows);

    // loop over all blocks
    for (size_t block = 0; block < nBlocks; ++block) {
        
        size_t rowStart;
        rowStart = block * blockRows;

        // fill the valarray with zero before getting the block of the data
        refSlc = 0;
        secSlc = 0;
        // get a block of reference and secondary slc
        referenceSLC.getBlock(refSlc, 0, rowStart, ncols, blockRows);
        secondarySLC.getBlock(secSlc, 0, rowStart, ncols, blockRows);
    
        
        //commaon azimuth band-pass filter 
        
        // upsample the refernce and secondary SLCs
        refSignal.upsample(refSlc, refSlcUpsampled, blockRows, nfft, oversample);
        secSignal.upsample(secSlc, secSlcUpsampled, blockRows, nfft, oversample);
        
        // Compute oversampled interferogram data
        for (size_t line=0; line < blockRows; ++line){
            for (size_t i=0; i<oversample*ncols; ++i){
                ifgramUpsampled[line*(oversample*ncols)+i] = refSlcUpsampled[line*(oversample*nfft)+i]*std::conj(secSlcUpsampled[line*(oversample*ncols)+i]);
            }
        }
        // Reclaim the extra oversample looks across
        for (size_t line=0; line < blockRows; ++line){
            for (size_t i=0; i<ncols; ++i){
                std::complex<float> sum =(0,0);
                for (size_t j=0; j<oversample; ++j)
                    sum += ifgramUpsampled[line*(nfft*oversample) + 2*i + j];
                ifgram[line*ncols + i] = sum;            
            }
        }

	// Take looks down (summing columns)
        
        // Take looks across (summing row blocks)
        
	// write data to file
        interferogram.setBlock(ifgram, 0, rowStart, ncols, blockRows);
    }
}



