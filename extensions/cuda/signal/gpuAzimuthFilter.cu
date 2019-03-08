// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#include <cstdio>
#include "gpuFilter.h"

#include "isce/signal/Filter.h"

using isce::cuda::signal::gpuAzimuthFilter;

template<class T>
gpuAzimuthFilter<T>::gpuAzimuthFilter()
{
    cufftType fft_type = (sizeof(T) == 8) ? CUFFT_Z2Z : CUFFT_C2C;
    this->_signal= gpuSignal<T>(fft_type);
}

template<class T>
gpuAzimuthFilter<T>::~gpuAzimuthFilter()
{
    if (this->_filter_set) {
        cudaFree(this->_d_filter);
    }
}

template<class T>
void gpuAzimuthFilter<T>::
initiateAzimuthFilter(std::valarray<std::complex<T>> &input,
        std::valarray<std::complex<T>> &spectrum,
        size_t ncols,
        size_t nrows) 
{
    // set FFT parameters
    this->_signal.azimuthFFT(ncols, nrows);
}

// construct filter on host then copy to device
template<class T>
void gpuAzimuthFilter<T>::
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
    // Pedestal-dependent frequency offset for transition region
    const double df = 0.5 * bandwidth * beta;
    // Compute normalization factor for preserving average power between input
    // data and filtered data. Assumes both filter and input signal have flat
    // spectra in the passband.
    //const double norm = std::sqrt(input_BW / BW);
    const double norm = 1.0;

    // we probably need to give next power of 2 ???
    //size_t nfft = 0;
    //this->_signal.nextPowerOfTwo(ncols, nfft);

    // Construct vector of frequencies
    std::valarray<double> frequency(ncols);
    
    isce::signal::Filter<float>::fftfreq(ncols, 1.0/prf, frequency);
    
    std::valarray<std::complex<T>> host_filter(ncols*nrows);

    // Loop over range bins
    for (int j = 0; j < ncols; ++j) {
        // Compute center frequency of common band
        const double fmid = 0.5 * (refDoppler.eval(j) + secDoppler.eval(j));

        // Compute filter
        for (size_t i = 0; i < frequency.size(); ++i) {

            // Get the absolute value of shifted frequency
            const double freq = std::abs(frequency[i] - fmid);

            // Passband
            if (freq <= (0.5 * bandwidth - df)) {
                host_filter[i*ncols+j] = std::complex<T>(norm, 0.0);

            // Transition region
            } else if (freq > (0.5 * bandwidth - df) && freq <= (0.5 * bandwidth + df)) {
                host_filter[i*ncols+j] = std::complex<T>(norm * 0.5 * 
                        (1.0 + std::cos(M_PI / (bandwidth*beta) * 
                        (freq - 0.5 * (1.0 - beta) * bandwidth))), 0.0);

            // Stop band
            } else {
                host_filter[i*ncols+j] = std::complex<T>(0.0, 0.0);
            }
        }
    }

    this->cpFilterHostToDevice(host_filter);

    this->_signal.azimuthFFT(ncols, nrows);
}

// DECLARATIONS
template class gpuAzimuthFilter<float>;
