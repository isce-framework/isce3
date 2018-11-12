
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
    isce::io::Raster inputSlc("../data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int length = inputSlc.length();

    // 
    int blockLength = length;

    // reserve memory for a block of data
    std::valarray<std::complex<float>> data(width*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<float>> range_spectrum(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<float>> invertData(width*blockLength);

    // a signal object
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

TEST(Signal, ForwardBackwardAzimuthFloat)
{
      // take a block of data, perform azimuth FFT and then iverse FFT and compare with original data
      isce::io::Raster inputSlc("../data/warped_envisat.slc.vrt");

      int width = inputSlc.width();
      int length = inputSlc.length();

      //
      int blockLength = length;

      // reserve memory for a block of data
      std::valarray<std::complex<float>> data(width*blockLength);

      // reserve memory for the spectrum of the block of data
      std::valarray<std::complex<float>> azimuth_spectrum(width*blockLength);

      // reserve memory for a block of data computed from inverse FFT
      std::valarray<std::complex<float>> invertData(width*blockLength);

      // a signal object
      isce::signal::Signal<float> sig;

      // create the forward and backward plans
      sig.forwardAzimuthFFT(data, azimuth_spectrum, width, blockLength, width, blockLength);
      sig.inverseAzimuthFFT(azimuth_spectrum, invertData, width, blockLength, width, blockLength);

      // read a block of data
      inputSlc.getBlock(data, 0, 0, width, blockLength);

      // forward fft transform
      sig.forward(data, azimuth_spectrum);

      // inverse fft transform
      sig.inverse(azimuth_spectrum, invertData);

      //normalize the result of inverse fft
      invertData /=width;

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


TEST(Signal, nfft)
{
    // This test is same as the previous test but with nfft used for FFT computation instead of number of columns
    isce::io::Raster inputSlc("../data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int length = inputSlc.length();

    int blockLength = length;
    
    // fft length for FFT computations 
    int nfft = 1024;

    // reserve memory for a block of data
    std::valarray<std::complex<float>> data(width*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<float>> range_spectrum(nfft*blockLength);

    // reserve memory for the block of data after inverse fft
    std::valarray<std::complex<float>> invertData(nfft*blockLength);

    // instantiate a signal object
    isce::signal::Signal<float> sig;

    // create the forward and backward plans
    sig.forwardRangeFFT(data, range_spectrum, width, blockLength, nfft, blockLength);
    sig.inverseRangeFFT(range_spectrum, invertData, nfft, blockLength, nfft, blockLength);


    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);

    // forward fft transform
    sig.forward(data, range_spectrum);

    // inverse fft transform
    sig.inverse(range_spectrum, invertData);

    //normalize the result of inverse fft
    invertData /=nfft;

    int blockSize = width*blockLength;
    std::complex<float> err(0.0, 0.0);
    bool Test = true;
    double max_err = 0.0;
    
    for (size_t line = 0; line<blockLength; ++line){
        for (size_t col = 0; col<width; ++col){
            err = invertData[line*nfft+col] - data[line*width+col];
            if (std::abs(err) > max_err){
                    max_err = std::abs(err);
            }
        }
    }

    ASSERT_LT(max_err, 1.0e-4);   

}



int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

