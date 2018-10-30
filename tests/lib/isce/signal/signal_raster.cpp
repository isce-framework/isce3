
#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <valarray>
#include <complex>
//#include <gtest/gtest.h>

#include "isce/signal/Signal.h"
#include "isce/io/Raster.h"

int main()
{
    
    isce::io::Raster inputSlc("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int length = inputSlc.length();

    std::cout << "width: " << width << std::endl;
    std::cout << "length: " << length << std::endl;

    int blockLength = length;
    int oversample = 2;

    int nfft = width;
    // reserve memory for a block of data and oversampled data
    std::valarray<std::complex<float>> data(width*blockLength);
    std::valarray<std::complex<float>> dataOv(width*blockLength*oversample);
    //std::valarray<std::complex<float>> range_spectrum(nfft*blockLength*oversample);
    std::valarray<std::complex<float>> range_spectrum(nfft*blockLength);
    std::valarray<std::complex<float>> range_spectrum_ov(nfft*blockLength*oversample);

    inputSlc.getBlock(data, 0, 0, width, blockLength);
    isce::signal::Signal<float> sig;
    sig.forwardRangeFFT(data, range_spectrum, width, blockLength, nfft, blockLength);
    sig.inverseRangeFFT(range_spectrum_ov, dataOv, nfft*oversample, blockLength, nfft*oversample, blockLength);

    sig.upsample(data, dataOv, blockLength, nfft, oversample);

    // Open raster for writing
    isce::io::Raster ovSlc("oversampled.slc", nfft*oversample, blockLength, 1,
                                          GDT_CFloat32, "ISCE");

    ovSlc.setBlock(dataOv, 0, 0, nfft*oversample, blockLength);

    //

    //std::valarray<std::complex<float>> range_spectrum(width*blockLength);
    //std::valarray<std::complex<float>> azimuth_spectrum(width*blockLength);
    //std::valarray<std::complex<float>> invertData(width*blockLength);

    // read a block of data
    //inputSlc.getBlock(data, 0, 0, width, blockLength);
    
    //std::cout << "call signal" << std::endl;
    //isce::signal::Signal<float> sig;
    //sig.forwardRangeFFT(data, range_spectrum, width, blockLength, width, blockLength);

    //sig.forwardAzimuthFFT(data, azimuth_spectrum, width, blockLength, width, blockLength);

    //sig.inverseRangeFFT(range_spectrum, invertData, width, blockLength, width, blockLength);
    

    //int i = width* 30 + 100;
    //int blockSize = width*blockLength;
    //std::complex<float> err(0.0, 0.0);
    //bool Test=true;
    /*for ( size_t i = 0; i < blockSize; ++i ) {
        err = invertData[i] - data[i];
        if (std::abs(err) > 1.0e-6){
            std::cout << "error: " << err  << std::endl;
            Test = false;
            break;
        }
    }

    if (Test)
        std::cout<< "PASSED" << std::endl;
    */
    //std::cout << " data: " << data[i] << std::endl;
    //std::cout << " inveretd data from range spectrum: " << invertData[i] << std::endl;
    //sig.inverseAzimuthFFT(azimuth_spectrum, invertData, width, blockLength, width, blockLength);
    //std::cout << " inverted data from azimuth spectrum : " << invertData[i] << std::endl;
    



    return (0);         

}


