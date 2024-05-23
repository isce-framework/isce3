//
// Author: Liang Yu
// Copyright 2019
//

#pragma once

#include "forward.h"
#include <isce3/core/forward.h>

#include <complex>
#include <valarray>
#include <thrust/complex.h>

#include "gpuSignal.h"

namespace isce3::cuda::signal {

// Definition of base class
template<class T>
class gpuFilter {
    public:
        gpuFilter() {};
        ~gpuFilter();

        /** Filter a signal in frequency domain*/
        void filter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum);

        /** Filter a signal in place on device */
        void filter(gpuSignal<T> &signal);

        /** Filter a signal in place on device */
        void filter(thrust::complex<T> *data);

        /** carry over from parent class. eliminate and use parent? */
        void writeFilter(size_t ncols, size_t nrows);

    protected:

        thrust::complex<T> *_d_filter;               // device unified memory pointer
        bool _filter_set = false;
        gpuSignal<T> _signal;
};

// Azimuth filter class derived from base class
template <class T>
class gpuAzimuthFilter : public gpuFilter<T> {
    public:
        gpuAzimuthFilter();
        ~gpuAzimuthFilter() {};

        /** constructs forward abd backward FFT plans for filtering a block of data in azimuth direction. */
        void initiateAzimuthFilter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum,
                size_t ncols,
                size_t nrows);

        void constructAzimuthCommonbandFilter(const isce3::core::LUT1d<double> & refDoppler,
                const isce3::core::LUT1d<double> & secDoppler,
                double bandwidth,
                double prf,
                double beta,
                //std::valarray<std::complex<T>> &signal,
                size_t ncols,
                size_t nrows);
};

// Range filter class derived from base class
template <class T>
class gpuRangeFilter : public gpuFilter<T> {
    public:
        gpuRangeFilter();
        ~gpuRangeFilter();

        // same name wrappers for filter init and construction functions with HostToDevice cp
        /** constructs forward abd backward FFT plans for filtering a block of data in range direction. */
        void initiateRangeFilter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum,
                size_t ncols,
                size_t nrows);

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

        void filterCommonRangeBand(T *d_refSlc, T *d_secSlc, T *range);

        size_t rangeFrequencyShiftMaxIdx(thrust::complex<T> *spectrum,
                int n_rows,
                int n_cols);

        void getPeakIndex(T *data, int data_lenth, size_t &peakIndex);

    private:
        double _wavelength;
        double _rangePixelSpacing;
        double _freqShift;
        double _rangeBandWidth;
        double _rangeSamplingFrequency;
        double _rangeBandwidth;
        T *_d_spectrumSum;
        bool _spectrumSum_set = false;
};

template<class T>
__global__ void phaseShift_g(thrust::complex<T> *slc, T *range, double pxlSpace, T conj, double wavelength, T wave_div, int n_elements);

template<class T>
__global__ void sumSpectrum_g(thrust::complex<T> *spectrum, T *spectrum_sum, int n_rows, int n_cols);

} // namespace isce3::cuda::signal