
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
    sig.forwardRangeFFT(data, range_spectrum, width, blockLength);
    sig.inverseRangeFFT(range_spectrum, invertData, width, blockLength);

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
      sig.forwardAzimuthFFT(data, azimuth_spectrum, width, blockLength);
      sig.inverseAzimuthFFT(azimuth_spectrum, invertData, width, blockLength);

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
    //width = 51;
    int blockLength = length;
    
    // fft length for FFT computations 
    int nfft = 1024;



    // reserve memory for a block of data with the size of nfft
    std::valarray<std::complex<float>> data(nfft*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<float>> range_spectrum(nfft*blockLength);

    // reserve memory for the block of data after inverse fft
    std::valarray<std::complex<float>> invertData(nfft*blockLength);

    data = 0;
    range_spectrum = 0;
    invertData = 0;

    // instantiate a signal object
    isce::signal::Signal<float> sig;

    // create the forward and backward plans
    sig.forwardRangeFFT(data, range_spectrum, nfft, blockLength);
    sig.inverseRangeFFT(range_spectrum, invertData, nfft, blockLength);


    // read a block of data
    std::valarray<std::complex<float>> dataLine(width);
    for (size_t line = 0; line<blockLength; ++line){
        inputSlc.getLine(dataLine, line);
        data[std::slice(line*nfft,width,1)] = dataLine; //[std::slice(0,width,0)];
    }
    //inputSlc.getBlock(data, 0, 0, width, blockLength);

    // forward fft transform
    sig.forward(data, range_spectrum);

    // inverse fft transform
    sig.inverse(range_spectrum, invertData);

    //normalize the result of inverse fft
    invertData /=nfft;

    /*isce::io::Raster outputSlc("secSlc.slc", nfft, length, 1, GDT_CFloat32, "ENVI");
    outputSlc.setBlock(invertData, 0 ,0 , nfft, length);
    */ 

    int blockSize = width*blockLength;
    std::complex<float> err(0.0, 0.0);
    bool Test = true;
    double max_err = 0.0;
    
    for (size_t line = 0; line<blockLength; line++){
        for (size_t col = 0; col<width; col++){
            err = invertData[line*nfft+col] - data[line*nfft+col];
            if (std::abs(err) > max_err){
                    max_err = std::abs(err);
            }
        }
    }

    ASSERT_LT(max_err, 1.0e-5);   

}

TEST(Signal, upsample)
  {
      // This test is same as the previous test but with nfft used for FFT computation instead of number of column  s
      isce::io::Raster inputSlc("../data/warped_envisat.slc.vrt");

      int width = inputSlc.width();
      int length = inputSlc.length();
      //width = 51;
      int blockLength = length;

      // fft length for FFT computations
      size_t nfft;

      // instantiate a signal object
      isce::signal::Signal<float> sig;
      sig.nextPowerOfTwo(width, nfft);

      // upsampling factor
      int oversample = 2;

      // reserve memory for a block of data with the size of nfft
      std::valarray<std::complex<float>> data(nfft*blockLength);

      // reserve memory for the spectrum of the block of data
      std::valarray<std::complex<float>> rangeSpectrum(nfft*blockLength);

      std::valarray<std::complex<float>> rangeSpectrumUpsampled(oversample*nfft*blockLength);

      // reserve memory for the block of data after inverse fft
      std::valarray<std::complex<float>> dataUpsampled(nfft*oversample*blockLength);

      data = 0;
      rangeSpectrum = 0;

      dataUpsampled = 0;
      rangeSpectrumUpsampled = 0;


      // create the forward and backward plans
      sig.forwardRangeFFT(data, rangeSpectrum, nfft, blockLength);
      sig.inverseRangeFFT(rangeSpectrumUpsampled, dataUpsampled, nfft*oversample, blockLength);


      // read a block of data
      std::valarray<std::complex<float>> dataLine(width);
      for (size_t line = 0; line<blockLength; ++line){
          inputSlc.getLine(dataLine, line);
          data[std::slice(line*nfft,width,1)] = dataLine; //[std::slice(0,width,0)];
      }

      sig.upsample(data, dataUpsampled, blockLength, nfft, oversample);

      /*isce::io::Raster outputSlc("slcUpsampled.slc", nfft*oversample, length, 1, GDT_CFloat32, "ENVI");
      outputSlc.setBlock(dataUpsampled, 0 ,0 , nfft*oversample, length);
      */

      // needs an evaluation of the upsampled SLC

}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

