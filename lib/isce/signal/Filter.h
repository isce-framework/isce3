// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#ifndef ISCE_SIGNAL_FILTER_H
#define ISCE_SIGNAL_FILTER_H

#include <cmath>
#include <valarray>

#include <isce/core/Constants.h>
#include <isce/io/Raster.h>
#include <isce/core/Poly2d.h>
#include "Signal.h"

// Declaration
namespace isce {
    namespace signal {
        template<class T>
        class Filter;
    }
}

template<class T>
class isce::signal::Filter {
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
                                       int nfft,
                                       std::valarray<std::complex<T>> &_filter1D);

        void constructRangeBandpassCosine(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             std::valarray<double>& frequency,
                             double beta,
                             std::valarray<std::complex<T>>& _filter1D);

        //T constructRangeCommonbandFilter();

        /** Construct azimuth common band filter*/
        void constructAzimuthCommonbandFilter(const isce::core::Poly2d & refDoppler,
                                const isce::core::Poly2d & secDoppler,
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

        /** Create a vector of frequencies*/
        void fftfreq(int N, double dt, std::valarray<double> &freq);

        /** Find the index of a specific frequency for a signal with a specific sampling rate*/
        void indexOfFrequency(double dt, int N, double f, int& n);

        void writeFilter(size_t ncols, size_t nrows);

    private:
        isce::signal::Signal<T> _signal;
        std::valarray<std::complex<T>> _filter;

};

#endif


