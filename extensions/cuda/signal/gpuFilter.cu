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
    if (_filter_set) {
        cudaFree(_d_filter);
    }
    if (_spectrumSum_set) {
        cudaFree(_d_spectrumSum);
    }
}

template<class T>
void gpuFilter<T>::
initiateRangeFilter(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &spectrum,
                        size_t ncols,
                        size_t nrows) 
{
    // malloc device memory for eventual max frequency search
    if (!_spectrumSum_set) {
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_spectrumSum), ncols*sizeof(T)*2));
        _spectrumSum_set = true;
    }

    // set FFT parameters
    cufftType fft_type = (sizeof(T) == 8) ? CUFFT_Z2Z : CUFFT_C2C;
    gpuSignal<T> _signal(fft_type);
    _signal.rangeFFT(ncols, nrows);
}

template<class T>
void gpuFilter<T>::
initiateAzimuthFilter(std::valarray<std::complex<T>> &input,
                           std::valarray<std::complex<T>> &spectrum,
                           size_t ncols,
                           size_t nrows) 
{
    // set FFT parameters
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

// construct filter on host then copy to device
template<class T>
void gpuFilter<T>::
constructRangeBandpassFilter(double rangeSamplingFrequency,
                            std::valarray<double> subBandCenterFrequencies,
                            std::valarray<double> subBandBandwidths,
                            size_t ncols,
                            size_t nrows,
                            std::string filterType)
{
    // TODO move filter construction to GPU?
    Filter<T>::constructRangeBandpassFilter(rangeSamplingFrequency,
                                            subBandCenterFrequencies,
                                            subBandBandwidths,
                                            ncols,
                                            nrows,
                                            filterType);

    cpuFilterHostToDevice();

    cufftType fft_type = (sizeof(T) == 8) ? CUFFT_Z2Z : CUFFT_C2C;
    gpuSignal<T> _signal(fft_type);
    _signal.rangeFFT(ncols, nrows);
}

// construct filter on host then copy to device
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
    cpuFilterHostToDevice();

    cufftType fft_type = (sizeof(T) == 8) ? CUFFT_Z2Z : CUFFT_C2C;
    gpuSignal<T> _signal(fft_type);
    _signal.rangeFFT(ncols, nrows);
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

    filter_g<<<grid, block>>>(signal.getDevicePtr(), _d_filter, signal.size());

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

    filter_g<<<grid, block>>>(data, _d_filter, _signal.size());

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

    filter_g<<<grid, block>>>(_signal.getDevicePtr(), _d_filter, signal.size());

    _signal.inverse();

    // copy signal to host
    _signal.dataToHost(signal);
}


/** range filters 2 signals in preparation for interferometry calculation
  calculates frequency shift to be applied 
  applies prerequisite phase shifts */
template<class T>
void gpuFilter<T>::
filterCommonRangeBand(T *d_refSlc, T *d_secSlc, T *range)
{
    auto n_elements = _signal.getNumElements();

    // determine block layout; set these in constructor since they're based on n_elements?
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_elements+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // apply full phase correction to both signals
    phaseShift_g<<<grid, block>>>(d_refSlc, range, _rangePixelSpacing, T(1.), _wavelength, T(1.), n_elements);
    phaseShift_g<<<grid, block>>>(d_secSlc, range, _rangePixelSpacing, T(-1.), _wavelength, T(1.), n_elements);

    auto ncols = _signal.getColumns();
    auto nrows = _signal.getRows();
    std::valarray<double> rangeFrequencies(ncols);
    Filter<T>::fftfreq(ncols, 1.0/_rangeSamplingFrequency, rangeFrequencies);

    // calculate frequency shift
    size_t refIdx = rangeFrequencyShiftMaxIdx(d_refSlc);
    size_t secIdx = rangeFrequencyShiftMaxIdx(d_secSlc);
    double frequencyShift = rangeFrequencies[refIdx] - rangeFrequencies[secIdx];

    std::valarray<double> filterCenterFrequency{0.0};
    std::valarray<double> filterBandwidth{_rangeBandwidth - frequencyShift};
    std::string filterType = "cosine";

    // TODO do this on GPU?
    constructRangeBandpassFilter(_rangeSamplingFrequency,
            filterCenterFrequency,
            filterBandwidth,
            ncols,
            nrows,
            filterType);

    //
    filter(d_refSlc);
    filter(d_secSlc);

    // apply half phase correction
    phaseShift_g<<<grid, block>>>(d_refSlc, range, _rangePixelSpacing, T(-1.), _wavelength, T(2.), n_elements);
    phaseShift_g<<<grid, block>>>(d_secSlc, range, _rangePixelSpacing, T(1.), _wavelength, T(2.), n_elements);
}


template<class T>
size_t gpuFilter<T>::
rangeFrequencyShiftMaxIdx(gpuComplex<T> *spectrum,
        double *rangeFrequencies,
        int n_elements)
{
    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_elements+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // sum spectrum along columns
    sumSpectrum_g<<<grid, block>>>(spectrum, _d_spectrumSum, n_elements);

    // copy to signal sums and find index of max value
    checkCudaErrors(cudaMemcpy(&_spectrumSum[0], _d_spectrumSum, n_elements*sizeof(T)*2, cudaMemcpyDeviceToHost));
    size_t idx = 0;
    getPeakIndex(_spectrumSum, idx);
    return idx;
}


template<class T>
void gpuFilter<T>::
getPeakIndex(std::valarray<float> data, size_t &peakIndex)
{
    size_t dataLength = data.size();
    peakIndex = 0;
    double peak = data[peakIndex];
    for (size_t i = 1; i< dataLength;  ++i){
        if (std::abs(data[i]) > peak){
            peak = data[i];
            peakIndex = i;
        }
    }
}


template<class T>
void gpuFilter<T>::
cpuFilterHostToDevice()
{
    if (!_filter_set) {
        size_t sz_filter = Filter<T>::_filter.size()*sizeof(T)*2;
        // allocate input
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_filter), sz_filter));
        // copy input
        checkCudaErrors(cudaMemcpy(_d_filter, &Filter<T>::_filter[0], sz_filter, cudaMemcpyHostToDevice));
        _filter_set = true;
    }
}


template<class T>
__global__ void phaseShift_g(gpuComplex<T> *slc, T *range, T pxlSpace, T conj, T wavelength, T wave_div, int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        T phase = 4.0*M_PI*pxlSpace*range[i]/wavelength;
        gpuComplex<T> complex_phase(cos(phase/wave_div), conj*sin(phase/wave_div));
        slc[i] *= complex_phase;
    }
}
template<>
__global__ void phaseShift_g<float>(gpuComplex<float> *slc, float *range, float pxlSpace, float conj, float wavelength, float wave_div, int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        float phase = 4.0*M_PI*pxlSpace*range[i]/wavelength;
        gpuComplex<float> complex_phase(cosf(phase/wave_div), conj*sinf(phase/wave_div));
        slc[i] *= complex_phase;
    }
}

template<>
__global__ void phaseShift_g<double>(gpuComplex<double> *slc, double *range, double pxlSpace, double conj, double wavelength, double wave_div, int n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
        double phase = 4.0*M_PI*pxlSpace*range[i]/wavelength;
        gpuComplex<double> complex_phase(cos(phase/wave_div), conj*sin(phase/wave_div));
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
