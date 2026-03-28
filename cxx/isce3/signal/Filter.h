// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#pragma once

#include "forward.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <valarray>

#include <isce3/core/Constants.h>
#include <isce3/io/Raster.h>
#include <isce3/core/LUT1d.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/forward.h>

#include <isce3/math/Bessel.h>
#include <isce3/math/Sinc.h>
#include "Signal.h"

// Declaration
namespace isce3 { namespace signal {
    /** Create a vector of frequencies*/
    void fftfreq(double dt, std::valarray<double> &freq);
}}

template<class T>
class isce3::signal::Filter {
    public:

        Filter() {};

        ~Filter() {};

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
                                       int fft_size,
                                       std::valarray<std::complex<T>> &_filter1D);

        /** Construct a cosine range band-pass filter for multiple bands*/
        void constructRangeBandpassCosine(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             std::valarray<double>& frequency,
                             double beta,
                             std::valarray<std::complex<T>>& _filter1D);

        /** Construct a kaiser range band-pass filter for multiple bands*/
        void constructRangeBandpassKaiser(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             int fft_size,
                             std::valarray<double>& frequency,
                             double beta,
                             std::valarray<std::complex<T>>& _filter1D);

        /** Construct the range common band filter*/
        void constructRangeCommonBandFilter(const double rangeSamplingFrequency,
                                        const double subBandCenterFrequency,
                                        const double subBandBandwidth,
                                        size_t ncols,
                                        size_t nrows,
                                        const std::string& filterType,
                                        const double windowParameter,
                                        const int maxFilterKernelSize = 256);

        /** Construct a kaiser range band-pass filter for one band
         * First constructs a time-domain FIR filter, then transforms to the frequency
         * domain.
         * Returns the frequency-domain filter coefficients.
         */
        void constructRangeCommonBandKaiserFilter(const double subBandCenterFrequency,
                             const double subBandBandwidth,
                             const double rangeSamplingFrequency,
                             const int fft_size,
                             const double beta,
                             std::valarray<std::complex<T>>& filter1D,
                             const int maxFilterKernelSize = 256);

        /** Construct azimuth common band filter*/
        double constructAzimuthCommonBandFilter(const std::valarray<double> & refDoppler,
                            const std::valarray<double> & secDoppler,
                            double bandwidth,
                            double prf,
                            double windowParameter,
                            size_t ncols,
                            size_t nrows,
                            std::string& filterType);

        /** Construct azimuth common band cosine filter with the doppler centroid compensation*/
        double constructAzimuthCommonBandCosineFilter(const std::valarray<double> & refDoppler,
                                const std::valarray<double> & secDoppler,
                                double bandwidth,
                                double prf,
                                double beta,
                                size_t ncols,
                                size_t nrows);

        /** Construct a kaiser range band-pass filter for one band
         * First constructs a time-domain FIR filter, then transforms to the frequency
         * domain.
         * Returns the frequency-domain filter coefficients.
         */
        double constructAzimuthCommonBandKaiserFilter(const std::valarray<double> & refDoppler,
                                const std::valarray<double> & secDoppler,
                                double bandwidth,
                                double prf,
                                double beta,
                                size_t ncols,
                                size_t nrows);

        /** Filter a signal in frequency domain*/
        void filter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum);

        /** Find the index of a specific frequency for a signal with a specific sampling rate*/
        static void indexOfFrequency(double dt, int N, double f, int& n);

        void writeFilter(size_t ncols, size_t nrows);

    public:
        /** Determine the filter window parameters for the Kaiser window method
         * @param[in] ripple Upper bound for the deviation (in dB) of the magnitude of the filter's frequency response from that of the desired filter (not including frequencies in any transition intervals).
         * @param[in] transition_width Width of transition region, normalized so that 1 corresponds to pi radians / sample.
         * @returns the length and the beta of the Kaiser window.
         */
        std::tuple<int, double> _kaiserord(const double ripple, const double transition_width);

        /** Compute the Kaiser parameter `beta`, given the attenuation 'ripple`
        * @param[in] ripple The desired attenuation in the stopband and maximum ripple in the passband, in dB.
        * @return beta
        */
        double _kaiser_beta(const double ripple);

        /** Return length, shape, and time samples for Kaiser filter design method
         * @param[in] stopatt Upper bound for the deviation (in dB) of the magnitude of the filter's frequency response from that of the desired filter (not including frequencies in any transition intervals).
         * @param[in] transition_width Width of transition region, normalized so that 1 corresponds to pi radians / sample.
         * @param[in] force_odd_len  Force to be odd length
         * @param[out] n the length of the Kaiser window.
         * @param[out] beta the beta parameter for the Kaiser window
         * @param[out] t time samples for Kaiser filter design method
         */
        void  _kaiser_design(const double stopatt,
                             const double transition_width,
                             const bool force_odd_len,
                             int &n,
                             double &beta,
                             std::valarray<double> &t);

        /** Impulse response (Fourier transform) of Kaiser window
         * @param[in] t time samples for Kaiser filter design method
         * @param[in] beta the beta parameter for the Kaiser window
         * @param[out] irf time samples for Kaiser filter design method
         */
        void  _kaiser_irf(const std::valarray<double> &t,
                          const double beta,
                          std::valarray<std::complex<T>> &irf);

        /**  Kaiser window with length n
         * @param[in] n the length of the Kaiser window.
         * @param[in] beta the beta parameter for the Kaiser window
         * @param[out] coeffs the Kaiser filter coefficients
         */
        void  _kaiser(const int n,
                      const double beta,
                      std::valarray<std::complex<T>> &coeffs);

        /**  Turn a low pass filter into a band pass filter by applying a phase ramp.
         * @param[in] low_pass_filter low pass filter
         * @param[in] fc the center frequency, in Hz divded by sampling rate, in Hz
         * @param[out] band_pass_filter band pass filter
         */
        void  _lowpass2bandpass(const std::valarray<std::complex<T>> &low_pass_filter,
                    const double fc,
                    std::valarray<std::complex<T>> &band_pass_filter);

        /**   Design a low pass filter having a passband shaped like a window using the Kaiser method
         * @param[in] bandwidth the signal bandwidth
         * @param[in] fs the sampling frequency
         * @param[in] beta the Kaiser window beta parameter.
         * @param[in] stopatt Upper bound for the deviation (in dB) of the magnitude of the filter's frequency response from that of the desired filter (not including frequencies in any transition intervals).
         * @param[in] transition_width transition width [0-1]
         * @param[in] force_odd_len Force to be odd length
         * @param[out] kaiser_window low bandpass kaiser window
         */
        void  _design_shaped_lowpass_filter(const double bandwidth,
                                            const double fs,
                                            const double window_shape,
                                            std::valarray<std::complex<T>> &coeffs,
                                            const double stopatt = 40.0,
                                            const double transition_width = 0.15,
                                            const bool force_odd_len = false);
    private:
        isce3::signal::Signal<T> _signal;
        std::valarray<std::complex<T>> _filter;

};
