
#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <valarray>
#include <complex>
#include <gtest/gtest.h>

#include "isce/signal/Signal.h"
#include "isce/io/Raster.h"

TEST(Signal, ForwardBackwardRangeFloat)
{
    // take a block of data, perform range FFT and then iverse FFT and compare with original data   
    isce::io::Raster inputSlc("/Users/fattahi/tools/ISCE3_forked/src/isce/tests/lib/isce/data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int length = inputSlc.length();

    std::cout << "width: " << width << std::endl;
    std::cout << "length: " << length << std::endl;

    int blockLength = length;

    // reserve memory for a block of data and spectrum
    std::valarray<std::complex<float>> data(width*blockLength);
    std::valarray<std::complex<float>> range_spectrum(width*blockLength);
    std::valarray<std::complex<float>> azimuth_spectrum(width*blockLength);
    std::valarray<std::complex<float>> invertData(width*blockLength);

    std::cout << "call signal" << std::endl;
    isce::signal::Signal<float> sig;
    // create the forward and backward plans
    sig.forwardRangeFFT(data, range_spectrum, width, blockLength, width, blockLength);
    sig.inverseRangeFFT(range_spectrum, invertData, width, blockLength, width, blockLength);


    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);
   
    // forward fft transform
    sig.forward(data, range_spectrum);

    // inverse fft transform
    sig.inverse(range_spectrum, invertData);

    //normalize the result of inverse fft 
    invertData /=width;

    //sig.forwardAzimuthFFT(data, azimuth_spectrum, width, blockLength, width, blockLength);

    


    //int i = width* 30 + 100;
    int blockSize = width*blockLength;
    std::complex<float> err(0.0, 0.0);
    bool Test = true;
    double max_err = 0.0;
    for ( size_t i = 0; i < blockSize; ++i ) {
        err = invertData[i] - data[i];
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-5);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

