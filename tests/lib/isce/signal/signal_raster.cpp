
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

    int blockLength = 100;

    // reserve memory for a block of data and spectrum
    std::valarray<std::complex<float>> data(width*blockLength);
    std::valarray<std::complex<float>> range_spectrum(width*blockLength);
    std::valarray<std::complex<float>> azimuth_spectrum(width*blockLength);
    std::valarray<std::complex<float>> invertData(width*blockLength);

    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);
    
    std::cout << "call signal" << std::endl;
    isce::signal::Signal<float> sig;
    sig.forwardRangeFFT(data, range_spectrum, width, blockLength, width, blockLength);

    sig.forwardAzimuthFFT(data, azimuth_spectrum, width, blockLength, width, blockLength);

    sig.inverseRangeFFT(range_spectrum, invertData, width, blockLength, width, blockLength);
    

    //int i = width* 30 + 100;
    int blockSize = width*blockLength;
    std::complex<float> err(0.0, 0.0);
    bool Test=true;
    for ( size_t i = 0; i < blockSize; ++i ) {
        err = invertData[i] - data[i];
        if (std::abs(err) > 1.0e-6){
            std::cout << "error: " << err  << std::endl;
            Test = false;
            break;
        }
    }

    if (Test)
        std::cout<< "PASSED" << std::endl;

    //std::cout << " data: " << data[i] << std::endl;
    //std::cout << " inveretd data from range spectrum: " << invertData[i] << std::endl;
    //sig.inverseAzimuthFFT(azimuth_spectrum, invertData, width, blockLength, width, blockLength);
    //std::cout << " inverted data from azimuth spectrum : " << invertData[i] << std::endl;
    



    return (0);         

}


