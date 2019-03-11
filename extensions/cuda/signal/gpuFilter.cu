// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#include "gpuFilter.h"
#include "isce/io/Raster.h"

#include "isce/cuda/helper_cuda.h"
#include "isce/cuda/helper_functions.h"

#define THRD_PER_BLOCK 1024 // Number of threads per block (should always %32==0)

using isce::cuda::signal::gpuFilter;

template<class T>
gpuFilter<T>::~gpuFilter()
{
    if (_filter_set) {
        cudaFree(_d_filter);
    }
}

// do all calculations in place with data stored on device within signal
template<class T>
void gpuFilter<T>::
filter(gpuSignal<T> &signal)
{
    signal.forward();

    auto n_signal_elements = signal.getNumElements();

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_signal_elements+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    filter_g<<<grid, block>>>(reinterpret_cast<gpuComplex<T> *>(signal.getDevicePtr()), 
            reinterpret_cast<gpuComplex<T> *>(&_d_filter), 
            n_signal_elements);

    signal.inverse();
}


// pass in device pointer to filter on
template<class T>
void gpuFilter<T>::
filter(gpuComplex<T> *data)
{
    _signal.forwardDevMem(reinterpret_cast<T *>(data));

    auto n_signal_elements = _signal.getNumElements();

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_signal_elements+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    filter_g<<<grid, block>>>(data, 
            reinterpret_cast<gpuComplex<T> *>(_d_filter), 
            n_signal_elements);

    _signal.inverseDevMem(reinterpret_cast<T *>(data));
}


// pass in host memory to copy to device to be filtered
// interim spectrum is saved as well
template<class T>
void gpuFilter<T>::
filter(std::valarray<std::complex<T>> &signal,
        std::valarray<std::complex<T>> &spectrum)
{
    _signal.dataToDevice(signal);
    _signal.forward();

    // save spectrum
    _signal.dataToHost(spectrum);

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((signal.size()+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    filter_g<<<grid, block>>>(reinterpret_cast<gpuComplex<T> *>(_signal.getDevicePtr()), 
            reinterpret_cast<gpuComplex<T> *>(&_d_filter), 
            signal.size());

    _signal.inverse();

    // copy signal to host
    _signal.dataToHost(signal);
}

template<class T>
void gpuFilter<T>::
cpFilterHostToDevice(std::valarray<std::complex<T>> &host_filter)
{
    if (!_filter_set) {
        size_t sz_filter = host_filter.size()*sizeof(gpuComplex<T>);
        // allocate input
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_filter), sz_filter));
        // copy input
        checkCudaErrors(cudaMemcpy(_d_filter, &host_filter[0], sz_filter, cudaMemcpyHostToDevice));
        _filter_set = true;
    }
}

template<class T>
void gpuFilter<T>::
writeFilter(size_t ncols, size_t nrows)
{
    std::valarray<std::complex<T>> filter;
    cpFilterHostToDevice(filter);
    isce::io::Raster filterRaster("filter.bin", ncols, nrows, 1, GDT_CFloat32, "ENVI");
    filterRaster.setBlock(filter, 0, 0, ncols, nrows);
}

template<class T>
__global__ void phaseShift_g(gpuComplex<T> *slc, 
        T *range, 
        double pxlSpace, 
        T conj, 
        double wavelength, 
        T wave_div, 
        int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        T phase = 4.0*M_PI*pxlSpace*range[i]/wavelength;
        gpuComplex<T> complex_phase(cos(phase/wave_div), conj*sin(phase/wave_div));
        slc[i] *= complex_phase;
    }
}

template<>
__global__ void phaseShift_g<float>(gpuComplex<float> *slc, 
        float *range, 
        double pxlSpace, 
        float conj, 
        double wavelength, 
        float wave_div, 
        int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        float phase = 4.0*M_PI*pxlSpace*range[i]/wavelength;
        gpuComplex<float> complex_phase(cosf(phase/wave_div), conj*sinf(phase/wave_div));
        slc[i] *= complex_phase;
    }
}

template<class T>
__global__ void filter_g(gpuComplex<T> *signal, gpuComplex<T> *filter, int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        signal[i] *= filter[i];
    }
}

template<class T>
__global__ void sumSpectrum_g(gpuComplex<T> *spectrum, T *spectrum_sum, int n_rows, int n_cols)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_cols) {
        for (int i_row = 0; i_row < n_rows; ++i_row) {
            spectrum_sum[i] += abs(spectrum[i_row*n_cols + i]);
        }
    }
}

// DECLARATIONS
template class gpuFilter<float>;

template __global__ void
sumSpectrum_g<float>(gpuComplex<float> *spectrum, float *spectrum_sum, int n_rows, int n_cols);
