// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#include "gpuFilter.h"

#include "isce/cuda/helper_cuda.h"
#include "isce/cuda/helper_functions.h"

#define THRD_PER_BLOCK 1024 // Number of threads per block (should always %32==0)

using isce::cuda::signal::gpuFilter;
using isce::signal::Filter;

template<class T>
gpuFilter<T>::~gpuFilter()
{
    if (_d_filter_set) {
        cudaFree(_d_filter);
    }
    if (_d_input_set) {
        cudaFree(_d_input);
    }
}

template<class T>
void gpuFilter<T>::
initiateRangeFilter(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &spectrum,
                        size_t ncols,
                        size_t nrows) {
    if (!_d_input_set) {
        size_t input_size = input.size()*sizeof(T)*2;
        // allocate input
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_input), input_size));
        // copy input
        checkCudaErrors(cudaMemcpy(_d_input, &input[0], input_size, cudaMemcpyHostToDevice));
        _d_input_set = true;
    }

    cufftType fft_type = (sizeof(T) == 8) ? CUFFT_Z2Z : CUFFT_C2C;
    gpuSignal<T> _signal(fft_type);
    _signal.rangeFFT(ncols, nrows);
}

template<class T>
void gpuFilter<T>::
initiateAzimuthFilter(std::valarray<std::complex<T>> &input,
                           std::valarray<std::complex<T>> &spectrum,
                           size_t ncols,
                           size_t nrows) {
    if (!_d_input_set) {
        size_t input_size = input.size()*sizeof(T)*2;
        // allocate input
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_input), input_size));
        // copy input
        checkCudaErrors(cudaMemcpy(_d_input, &input[0], input_size, cudaMemcpyHostToDevice));
        _d_input_set = true;
    }

    cufftType fft_type = (sizeof(T) == 8) ? CUFFT_Z2Z : CUFFT_C2C;
    gpuSignal<T> _signal(fft_type);
    _signal.azimuthFFT(ncols, nrows);
}

template<class T>
void gpuFilter<T>::
constructRangeBandpassFilter(double rangeSamplingFrequency,
                            std::valarray<double> subBandCenterFrequencies,
                            std::valarray<double> subBandBandwidths,
                            std::valarray<std::complex<T>> &signal,
                            std::valarray<std::complex<T>> &spectrum,
                            size_t ncols,
                            size_t nrows,
                            std::string filterType)

{
    constructRangeBandpassFilter(rangeSamplingFrequency,
                                subBandCenterFrequencies,
                                subBandBandwidths,
                                ncols,
                                nrows,
                                filterType);

    cufftType fft_type = (sizeof(T) == 8) ? CUFFT_Z2Z : CUFFT_C2C;
    gpuSignal<T> _signal(fft_type);
    _signal.rangeFFT(ncols, nrows);
}

template<class T>
void gpuFilter<T>::
constructRangeBandpassFilter(double rangeSamplingFrequency,
                            std::valarray<double> subBandCenterFrequencies,
                            std::valarray<double> subBandBandwidths,
                            size_t ncols,
                            size_t nrows,
                            std::string filterType)
{
    // construct range bandpass on CPU then copy to GPU
    // TODO move filter construction to GPU
    Filter<T>::constructRangeBandpassFilter(rangeSamplingFrequency,
                                            subBandCenterFrequencies,
                                            subBandBandwidths,
                                            ncols,
                                            nrows,
                                            filterType);

    cpFilterToDevice();
}

template<class T>
void gpuFilter<T>::
constructAzimuthCommonbandFilter(const isce::core::LUT1d<double> & refDoppler,
                                const isce::core::LUT1d<double> & secDoppler,
                                double bandwidth,
                                double prf,
                                double beta,
                                std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows)
{
    Filter<T>::constructAzimuthCommonbandFilter(refDoppler,
                                            secDoppler,
                                            bandwidth,
                                            prf,
                                            beta,
                                            signal,
                                            spectrum,
                                            ncols,
                                            nrows);
    cpFilterToDevice();
}


template<class T>
__global__ void filter_g(gpuComplex<T> *signal, gpuComplex<T> *filter, int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        signal[i] *= filter[i];
    }
}


// keep everything in place on device
template<class T>
void gpuFilter<T>::
filter(gpuSignal<T> &signal)
{
    signal.forward();

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((signal.size()+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    filter_g<<<grid, block>>>(signal.getDevicePtr(), &_d_filter, signal.size());

    signal.inverse();
}


// keep everything in place on device
template<class T>
void gpuFilter<T>::
filter(gpuComplex<T> *data)
{
    _signal.forward(data);

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((_signal.size()+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    filter_g<<<grid, block>>>(data, &_d_filter, _signal.size());

    _signal.inverse(data);
}


template<class T>
void gpuFilter<T>::
filter(std::valarray<std::complex<T>> &signal,
        std::valarray<std::complex<T>> &spectrum)
{
    _signal.dataToDevice(signal);
    _signal.forward();

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((signal.size()+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    filter_g<<<grid, block>>>(_signal.getDevicePtr(), &_d_filter, signal.size());

    _signal.inverse();

    // copy signal to host
    _signal.dataToHost(signal);
}

template<class T>
void gpuFilter<T>::
cpFilterToDevice()
{
    if (!_d_filter_set) {
        size_t sz_filter = Filter<T>::_filter.size()*sizeof(T)*2;
        // allocate input
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_filter), sz_filter));
        // copy input
        checkCudaErrors(cudaMemcpy(_d_filter, &Filter<T>::_filter[0], sz_filter, cudaMemcpyHostToDevice));
        _d_filter_set = true;
    }
}
