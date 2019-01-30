#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <valarray>
#include <complex>
#include <cufft.h>
#include <cufftXt.h>
#include <gtest/gtest.h>

#include "isce/signal/Signal.h"
#include "isce/io/Raster.h"

#include "isce/cuda/signal/gpuSignal.h"

#include "isce/signal/fftw3cxx.h"

// debug
#include <fstream>

using isce::cuda::signal::gpuSignal;

TEST(gpuSignal, ForwardBackwardRangeFloat)
{
    // take a block of data, perform range FFT and then iverse FFT and compare with original data   
    isce::io::Raster inputSlc("../../../../../lib/isce/data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int length = inputSlc.length();

    // 
    int blockLength = length;

    // reserve memory for a block of data
    std::valarray<std::complex<float>> data(width*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<float>> range_spectrum(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<float>> inverted_data(width*blockLength);

    // a signal object
    gpuSignal<float> sig(CUFFT_C2C);

    // create the plan
    sig.rangeFFT(width, blockLength);
    
    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);
   
    // forward fft transform
    sig.forwardC2C(data, range_spectrum);

    // inverse fft transform
    sig.inverseC2C(range_spectrum, inverted_data);

    //normalize the result of inverse fft 
    inverted_data /=width;

    int blockSize = width*blockLength;
    std::complex<float> err(0.0, 0.0);
    bool Test = true;
    double max_err = 0.0;
    for ( size_t i = 0; i < blockSize; ++i ) {
        err = inverted_data[i] - data[i];
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-4);
}

TEST(gpuSignal, ForwardBackwardAzimuthFloat)
{
    // take a block of data, perform range FFT and then iverse FFT and compare with original data   
    isce::io::Raster inputSlc("../../../../../lib/isce/data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int length = inputSlc.length();

    // 
    int blockLength = length;

    // reserve memory for a block of data
    std::valarray<std::complex<float>> data(width*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<float>> range_spectrum(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<float>> inverted_data(width*blockLength);

    // a signal object
    gpuSignal<float> sig(CUFFT_C2C);

    // create the plan
    sig.azimuthFFT(width, blockLength);
    
    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);
   
    // forward fft transform
    sig.forwardC2C(data, range_spectrum);

    // save outputs

    // inverse fft transform
    sig.inverseC2C(range_spectrum, inverted_data);

    //normalize the result of inverse fft 
    inverted_data /=width;

    int blockSize = width*blockLength;
    std::complex<float> err(0.0, 0.0);
    bool Test = true;
    double max_err = 0.0;
    for ( size_t i = 0; i < blockSize; ++i ) {
        err = inverted_data[i] - data[i];
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-4);
}

TEST(gpuSignal, nfft)
{
    // This test is same as the previous test but with nfft used for FFT computation instead of number of columns
    isce::io::Raster inputSlc("../../../../../lib/isce/data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int length = inputSlc.length();
    int blockLength = length;
    
    // fft length for FFT computations 
    int nfft = 512;

    // reserve memory for a block of data
    std::valarray<std::complex<float>> data(nfft*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<float>> range_spectrum(nfft*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<float>> inverted_data(nfft*blockLength);

    // read a block of data
    std::valarray<std::complex<float>> dataLine(width);
    for (size_t line = 0; line<blockLength; ++line){
        inputSlc.getLine(dataLine, line);
        data[std::slice(line*nfft,width,1)] = dataLine;
    }

    // a signal object
    gpuSignal<float> sig(CUFFT_C2C);

    // create the forward and backward plans
    sig.rangeFFT(nfft, blockLength);

    // forward fft transform
    sig.forwardC2C(data, range_spectrum);

    // inverse fft transform
    sig.inverseC2C(range_spectrum, inverted_data);

    //normalize the result of inverse fft
    inverted_data /= nfft;

    int blockSize = width*blockLength;
    std::complex<float> err(0.0, 0.0);
    bool Test = true;
    double max_err = 0.0;
    for ( size_t i = 0; i < blockSize; ++i ) {
        err = inverted_data[i] - data[i];
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-4);
}

TEST(gpuSignal, nfftDouble)
{
    int width = 100; //16;
    int length = 1;
    int blockLength = length;

    // fft length for FFT computations
    size_t nfft;

    // instantiate a signal object
    gpuSignal<double> sig(CUFFT_Z2Z);
    sig.nextPowerOfTwo(width, nfft);

    // reserve memory for a block of data with the size of nfft
    std::valarray<std::complex<double>> data(nfft);
    std::valarray<std::complex<double>> spec(nfft);
    std::valarray<std::complex<double>> inverted_data(nfft);

    for (size_t i=0; i<width; ++i){
        double phase = std::sin(10*M_PI*i/width);
        data[i] = std::complex<double> (std::cos(phase), std::sin(phase));
    }

    // create the forward and backward plans
    sig.rangeFFT(nfft, 1);

    // forward fft transform
    sig.forwardZ2Z(data, spec);

    // inverse fft transform
    sig.inverseZ2Z(spec, inverted_data);

    //normalize the result of inverse fft
    inverted_data /= nfft;

    int blockSize = width*blockLength;
    std::complex<float> err(0.0, 0.0);
    bool Test = true;
    double max_err = 0.0;
    for ( size_t i = 0; i < blockSize; ++i ) {
        err = inverted_data[i] - data[i];
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-4);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
