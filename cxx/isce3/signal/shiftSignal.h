// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-

#pragma once

#include "forward.h"

#include <cmath>
#include <valarray>
#include "Filter.h"
#include <isce3/core/Utilities.h>


namespace isce3 {
    namespace signal {

        /**
         *\brief shift a signal by constant offsets in x (columns) or y (rows) directions 
         */
        template<typename T, typename U>
        void shiftSignal(std::valarray<T> & data,
            std::valarray<T> & dataShifted,
            std::valarray<std::complex<U>> & spectrum,
            size_t ncols, size_t nrows,
            const double shiftX, const double shiftY,
            isce3::signal::Signal<U> & sigObj);

        /**
         *\brief shift a signal by constant offsets in x (columns) or y (rows) directions
         */
        template<typename T, typename U>
        void shiftSignal(std::valarray<T> & data,
            std::valarray<T> & dataShifted,
            std::valarray<std::complex<U>> & spectrum,
            std::valarray<std::complex<U>> & phaseRamp,
            isce3::signal::Signal<U> & sigObj);
        
        /**
         *\brief compute the impact of a constant range pixel shift in frequency domain
         */
        template<typename T>
        void frequencyResponseRange(size_t ncols, size_t nrows, 
            const double shift,
            std::valarray<std::complex<T>> & shiftImpact);
        
	/**
         *\brief compute the impact of a constant azimuth pixel shift in frequency domain
         */
        template<typename T>
        void frequencyResponseAzimuth(size_t ncols, size_t nrows, 
            const double shift,
            std::valarray<std::complex<T>> & shiftImpact);
    }
}
