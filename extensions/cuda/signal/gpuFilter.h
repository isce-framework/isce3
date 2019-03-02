// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#ifndef ISCE_CUDA_SIGNAL_FILTER_H
#define ISCE_CUDA_SIGNAL_FILTER_H

#include <complex>
#include <valarray>

#include "isce/signal/Filter.h"
#include "gpuSignal.h"
#include "isce/cuda/core/gpuComplex.h"

using isce::signal::Filter;
using isce::cuda::core::gpuComplex;
using isce::cuda::signal::gpuSignal;

// Declaration
namespace isce {
    namespace cuda {
        namespace signal {
            template<class T>
            class gpuFilter;
        }
    }
}

// Definition
template<class T>
class isce::cuda::signal::gpuFilter : public Filter<T>{

    public:
        // Default constructor
        gpuFilter() {};
        ~gpuFilter();

        // same name wrappers for filter init and construction functions with HostToDevice cp
        /** constructs forward abd backward FFT plans for filtering a block of data in range direction. */
        void initiateRangeFilter(std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows);

        /** constructs forward abd backward FFT plans for filtering a block of data in azimuth direction. */
        void initiateAzimuthFilter(std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows);

        /** Sets an existing filter to be used by the filter object*/
        //void setFilter(std::valarray<std::complex<T>>);

        /** Construct range band-pass filter*/
        void constructRangeBandpassFilter(double rangeSamplingFrequency,
                                        std::valarray<double> subBandCenterFrequencies,
                                        std::valarray<double> subBandBandwidths,
                                        std::valarray<std::complex<T>> &signal,
                                        std::valarray<std::complex<T>> &spectrum,
                                        size_t ncols,
                                        size_t nrows,
                                        std::string filterType);

        void constructRangeBandpassFilter(double rangeSamplingFrequency,
                                        std::valarray<double> subBandCenterFrequencies,
                                        std::valarray<double> subBandBandwidths,
                                        size_t ncols,
                                        size_t nrows,
                                        std::string filterType);

        /** Construct a box car range band-pass filter for multiple bands*/
        void constructRangeBandpassBoxcar(std::valarray<double> subBandCenterFrequencies,
                                       std::valarray<double> subBandBandwidths,
                                       double dt,
                                       int nfft,
                                       std::valarray<std::complex<T>> &_filter1D);

        void constructRangeBandpassCosine(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             std::valarray<double>& frequency,
                             double beta,
                             std::valarray<std::complex<T>>& _filter1D);

        /** Construct azimuth common band filter*/
        void constructAzimuthCommonbandFilter(const isce::core::LUT1d<double> & refDoppler,
                                const isce::core::LUT1d<double> & secDoppler,
                                double bandwidth,
                                double prf,
                                double beta,
                                std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows);

        /** Filter a signal in frequency domain*/
        void filter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum);

        /** Filter a signal in place on device */
        void filter(gpuSignal<T> &signal);

        /** Filter a signal in place on device */
        void filter(gpuComplex<T> *data);

        /** carry over from parent class. eliminate and use parent? */
        void writeFilter(size_t ncols, size_t nrows);

        void cpuFilterHostToDevice();

        // vvv eventually make part of derived range filter class vvv
        void filterCommonRangeBand(T *d_refSlc, T *secSlc, T *range);

        size_t rangeFrequencyShiftMaxIdx(gpuComplex<T> *spectrum,
                double *rangeFrequencies,
                int n_elements);

        void getPeakIndex(std::valarray<float> data, size_t &peakIndex);

    private:
        // device memory pointer
        T *_d_filter;
        bool _filter_set;
        gpuSignal<T> _signal;

        // vvv eventually make part of derived range filter class vvv
        double _wavelength;
        double _rangePixelSpacing;
        double _freqShift;
        double _rangeBandWidth;
        double _rangeSamplingFrequency;
        double _rangeBandwidth;
        T *_d_spectrumSum;
        bool _spectrumSum_set;
        std::valarray<std::complex<T>> _spectrumSum;
};

template<class T>
__global__ void phaseShift_g(gpuComplex<T> *slc, T *range, T pxlSpace, T conj, T wavelength, T wave_div, int n_elements);

template<class T>
__global__ void filter_g(gpuComplex<T> *signal, gpuComplex<T> *filter, int n_elements);

#endif

// end of file
