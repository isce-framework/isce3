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
#include <thrust/complex.h>
#include <gtest/gtest.h>

#include "isce/signal/Signal.h"
#include "isce/io/Raster.h"

#include "isce/cuda/signal/gpuSignal.h"

using isce::cuda::signal::gpuSignal;

TEST(gpuSignal, ForwardBackwardRangeFloat)
{
    // take a block of data, perform range FFT and then iverse FFT and compare with original data
    isce::io::Raster inputSlc("../../../../lib/isce/data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int blockLength = inputSlc.length();
    float *d_data;

    // reserve memory for a block of data
    std::valarray<std::complex<float>> data(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<float>> inverted_data(width*blockLength);

    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);

    // copy data to device
    cudaError_t oops;
    size_t data_sz = width * blockLength * sizeof(thrust::complex<float>);
    oops = cudaMalloc(reinterpret_cast<void **>(&d_data), data_sz);
    oops = cudaMemcpy(d_data, &data[0], data_sz, cudaMemcpyHostToDevice);

    // a signal object
    gpuSignal<float> sig(CUFFT_C2C);

    // create the plan
    sig.rangeFFT(width, blockLength);

    sig.forwardDevMem(d_data);
    sig.inverseDevMem(d_data);

    cudaMemcpy(&inverted_data[0], d_data, data_sz, cudaMemcpyDeviceToHost);

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

TEST(gpuSignal, ForwardBackwardRangeDouble)
{
    // take a block of data, perform range FFT and then iverse FFT and compare with original data
    isce::io::Raster inputSlc("../../../../lib/isce/data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int blockLength = inputSlc.length();
    double *d_data;

    // reserve memory for a block of data
    std::valarray<std::complex<double>> data(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<double>> inverted_data(width*blockLength);

    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);

    // copy data to device
    cudaError_t oops;
    size_t data_sz = width * blockLength * sizeof(thrust::complex<double>);
    oops = cudaMalloc(reinterpret_cast<void **>(&d_data), data_sz);
    oops = cudaMemcpy(d_data, &data[0], data_sz, cudaMemcpyHostToDevice);

    // a signal object
    gpuSignal<double> sig(CUFFT_Z2Z);

    // create the plan
    sig.rangeFFT(width, blockLength);

    sig.forwardDevMem(d_data);
    sig.inverseDevMem(d_data);

    cudaMemcpy(&inverted_data[0], d_data, data_sz, cudaMemcpyDeviceToHost);

    //normalize the result of inverse fft
    inverted_data /=width;

    int blockSize = width*blockLength;
    std::complex<double> err(0.0, 0.0);
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
    isce::io::Raster inputSlc("../../../../lib/isce/data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int blockLength = inputSlc.length();
    float *d_data;

    // reserve memory for a block of data
    std::valarray<std::complex<float>> data(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<float>> inverted_data(width*blockLength);

    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);

    // copy data to device
    cudaError_t oops;
    size_t data_sz = width * blockLength * sizeof(thrust::complex<float>);
    oops = cudaMalloc(reinterpret_cast<void **>(&d_data), data_sz);
    oops = cudaMemcpy(d_data, &data[0], data_sz, cudaMemcpyHostToDevice);

    // a signal object
    gpuSignal<float> sig(CUFFT_C2C);

    // create the plan
    sig.azimuthFFT(width, blockLength);

    sig.forwardDevMem(d_data);
    sig.inverseDevMem(d_data);

    cudaMemcpy(&inverted_data[0], d_data, data_sz, cudaMemcpyDeviceToHost);

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

TEST(gpuSignal, ForwardBackwardAzimuthDouble)
{
    // take a block of data, perform range FFT and then iverse FFT and compare with original data
    isce::io::Raster inputSlc("../../../../lib/isce/data/warped_envisat.slc.vrt");

    int width = inputSlc.width();
    int blockLength = inputSlc.length();
    double *d_data;

    // reserve memory for a block of data
    std::valarray<std::complex<double>> data(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<double>> inverted_data(width*blockLength);

    // read a block of data
    inputSlc.getBlock(data, 0, 0, width, blockLength);

    // copy data to device
    cudaError_t oops;
    size_t data_sz = width * blockLength * sizeof(thrust::complex<double>);
    oops = cudaMalloc(reinterpret_cast<void **>(&d_data), data_sz);
    oops = cudaMemcpy(d_data, &data[0], data_sz, cudaMemcpyHostToDevice);

    // a signal object
    gpuSignal<double> sig(CUFFT_Z2Z);

    // create the plan
    sig.azimuthFFT(width, blockLength);

    sig.forwardDevMem(d_data);
    sig.inverseDevMem(d_data);

    cudaMemcpy(&inverted_data[0], d_data, data_sz, cudaMemcpyDeviceToHost);

    //normalize the result of inverse fft
    inverted_data /=width;

    int blockSize = width*blockLength;
    std::complex<double> err(0.0, 0.0);
    bool Test = true;
    double max_err = 0.0;
    for ( size_t i = 0; i < blockSize; ++i ) {
        err = inverted_data[i] - data[i];
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    ASSERT_LT(max_err, 1.0e-9);
}

TEST(gpuSignal, upsampleFloat)
{
    int width = 100;
    int length = 1;
    int blockLength = length;

    // fft length for FFT computations
    size_t nfft;

    //sig.nextPowerOfTwo(width, nfft);
    nfft = width;
    // upsampling factor
    int oversample = 2;

    // reserve memory for a block of data with the size of nfft
    std::valarray<std::complex<float>> slc(nfft);
    std::valarray<std::complex<float>> slcU(nfft*oversample);

    for (size_t i=0; i<width; ++i){
        float phase = std::sin(10*M_PI*i/width);
        slc[i] = std::complex<float> (std::cos(phase), std::sin(phase));
    }

    // instantiate a signal object
    gpuSignal<float> sig_lo_res(CUFFT_C2C);
    gpuSignal<float> sig_hi_res(CUFFT_C2C);

    // create plans
    sig_lo_res.rangeFFT(nfft, 1);
    sig_hi_res.rangeFFT(oversample*nfft, 1);

    upsample(sig_lo_res,
            sig_hi_res,
            slc,
            slcU);

    // Check if the original smaples have the same phase in the signal before and after upsampling
    float max_err = 0.0;
    float err = 0.0;
    for (size_t col = 0; col<width; col++){
        err = std::arg(slc[col] * std::conj(slcU[oversample*col]));
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    float max_err_u = 0.0;
    float err_u;
    float step = 1.0/oversample;
    std::complex<float> cpxData;
    for (size_t col = 0; col<width*oversample; col++){
        float i = col*step;
        float phase = std::sin(10*M_PI*i/(width));
        cpxData = std::complex<float> (std::cos(phase), std::sin(phase));
        err_u = std::arg(cpxData * std::conj(slcU[col]));
        if (std::abs(err_u) > max_err_u){
              max_err_u = std::abs(err_u);
        }
    }

    ASSERT_LT(max_err, 1.0e-6);
    ASSERT_LT(max_err_u, 1.0e-6);
}

TEST(gpuSignal, upsampleDouble)
{
    int width = 100;
    int length = 1;
    int blockLength = length;

    // fft length for FFT computations
    size_t nfft;

    //sig.nextPowerOfTwo(width, nfft);
    nfft = width;
    // upsampling factor
    int oversample = 2;

    // reserve memory for a block of data with the size of nfft
    std::valarray<std::complex<double>> slc(nfft);
    std::valarray<std::complex<double>> slcU(nfft*oversample);

    for (size_t i=0; i<width; ++i){
        double phase = std::sin(10*M_PI*i/width);
        slc[i] = std::complex<double> (std::cos(phase), std::sin(phase));
    }

    // instantiate a signal object
    gpuSignal<double> sig_lo_res(CUFFT_Z2Z);
    gpuSignal<double> sig_hi_res(CUFFT_Z2Z);

    // create plans
    sig_lo_res.rangeFFT(nfft, 1);
    sig_hi_res.rangeFFT(oversample*nfft, 1);

    upsample(sig_lo_res,
            sig_hi_res,
            slc,
            slcU);

    // Check if the original smaples have the same phase in the signal before and after upsampling
    double max_err = 0.0;
    double err = 0.0;
    for (size_t col = 0; col<width; col++){
        err = std::arg(slc[col] * std::conj(slcU[oversample*col]));
        if (std::abs(err) > max_err){
            max_err = std::abs(err);
        }
    }

    double max_err_u = 0.0;
    double err_u;
    double step = 1.0/oversample;
    std::complex<double> cpxData;
    for (size_t col = 0; col<width*oversample; col++){
        double i = col*step;
        double phase = std::sin(10*M_PI*i/(width));
        cpxData = std::complex<double> (std::cos(phase), std::sin(phase));
        err_u = std::arg(cpxData * std::conj(slcU[col]));
        if (std::abs(err_u) > max_err_u){
              max_err_u = std::abs(err_u);
        }
    }

    ASSERT_LT(max_err, 1.0e-14);
    ASSERT_LT(max_err_u, 1.0e-9);
}

TEST(gpuSignal, FFT2D)
{
    int width = 12;
    int length = 10;
    double *d_data;

    //
    int blockLength = length;

    // reserve memory for a block of data
    std::valarray<std::complex<double>> data(width*blockLength);

    // reserve memory for the spectrum of the block of data
    std::valarray<std::complex<double>> spectrum(width*blockLength);

    // reserve memory for a block of data computed from inverse FFT
    std::valarray<std::complex<double>> invertData(width*blockLength);

    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            data[i*width + j] = std::complex<double> (std::cos(i*j), std::sin(i*j));
        }
    }

    // copy data to device
    cudaError_t oops;
    size_t data_sz = width * blockLength * sizeof(thrust::complex<double>);
    oops = cudaMalloc(reinterpret_cast<void **>(&d_data), data_sz);
    oops = cudaMemcpy(d_data, &data[0], data_sz, cudaMemcpyHostToDevice);

    // a signal object
    gpuSignal<double> sig(CUFFT_Z2Z);

    // create plan
    sig.FFT2D(width, blockLength);

    sig.forwardDevMem(d_data);
    sig.inverseDevMem(d_data);

    cudaMemcpy(&invertData[0], d_data, data_sz, cudaMemcpyDeviceToHost);

    invertData /= width*length;

    double max_err = 0.0;
    double err = 0.0;
    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            err = std::abs(data[i*width + j] - invertData[i*width + j]);
            if (err > max_err)
                max_err = err;
        }
    }

    ASSERT_LT(max_err, 1.0e-12);
}

TEST(gpuSignal, realDoubleDataFFT)
{
    int width = 120;
    int length = 100;

    int blockLength = length;

    // reserve memory for a block of data
    double *data = new double[width*blockLength];

    // reserve memory for the spectrum of the block of data
    std::complex<double> *spectrum = new std::complex<double>[width*blockLength];

    // reserve memory for a block of data computed from inverse FFT
    double *invertData = new double[width*blockLength];

    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            data[i*width + j] = i+j;
        }
    }

    // a signal objects
    gpuSignal<double> sig_D2Z(CUFFT_D2Z);
    gpuSignal<double> sig_Z2D(CUFFT_Z2D);

    // make plans
    sig_D2Z.FFT2D(width, blockLength);
    sig_Z2D.FFT2D(width, blockLength);

    // forward and inverse transform
    sig_D2Z.forwardD2Z(data, spectrum);
    sig_Z2D.inverseZ2D(spectrum, invertData);

    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            invertData[i*width + j] = i+j;
        }
    }

    double max_err_2DFFT = 0.0;
    double err = 0.0;
    for (size_t i = 0; i< length; ++i){
        for (size_t j = 0; j< width; ++j){
            err = std::abs(data[i*width + j] - invertData[i*width + j]);
            if (err > max_err_2DFFT)
                max_err_2DFFT = err;
        }
    }

    ASSERT_LT(max_err_2DFFT, 1.0e-12);
}

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
