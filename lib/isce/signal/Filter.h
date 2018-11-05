// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
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

        T constructAzimuthCommonbandFilter(const isce::core::Poly2d & refDoppler,
    				const isce::core::Poly2d & secDoppler,
    				double bandwidth,
    				double pulseRepetitionInterval,
    				double beta,
                                std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows);

        T filterSpectrum(std::valarray<std::complex<T>> &signal,
                         std::valarray<std::complex<T>> &spectrum);

        //T constructRangeCommonbandFilter();

        T constructRangeBandPassFilter(double rangeSamplingFrequency, 
                                        std::valarray<double> subBandCenterFrequencies, 
                                        std::valarray<double> subBandBandwidths,
                                        int ncols, int nrows);

        T fftfreq(int N, double dt, std::valarray<double> &freq);

        T indexOfFrequency(double dt, int N, double f, int& n);

    private:
        isce::signal::Signal<T> _signalAzimuth;
        isce::signal::Signal<T> _signalRange;
        std::valarray<std::complex<T>> _filter;
};

#endif


