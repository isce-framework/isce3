#ifndef ISCE_LIB_SHIFTSIGNAL_H
#define ISCE_LIB_SHIFTSIGNAL_H

#include <cmath>
#include <valarray>
#include "Signal.h"
#include "Filter.h"

namespace isce {
    namespace signal {

        template<typename T>
        void shiftSignal(std::valarray<T> & data,
            std::valarray<T> & dataShifted,
            std::valarray<std::complex<T>> & spectrum,
            size_t ncols, size_t nrows,
            const double shiftX, const double shiftY,
            isce::signal::Signal<T> & sigObj);

        template<typename T>
        void shiftSignal(std::valarray<std::complex<T>> & data,
                std::valarray<std::complex<T>> & dataShifted,
                std::valarray<std::complex<T>> & spectrum,
                size_t ncols, size_t nrows,
                const double shiftX, const double shiftY, 
                isce::signal::Signal<T> & sigObj);

        template<typename T>
        void shiftSignal(std::valarray<T> & data,
                std::valarray<T> & dataShifted,
                std::valarray<std::complex<T>> & spectrum,
                size_t ncols, size_t nrows,
                const double shift, isce::signal::Signal<T> & sigObj);

        template<typename T>
        void shiftSignal(std::valarray<std::complex<T>> & data,
                std::valarray<std::complex<T>> & dataShifted,
                std::valarray<std::complex<T>> & spectrum,
                size_t ncols, size_t nrows,
                const double shift, isce::signal::Signal<T> & sigObj);

        template<typename T>
        void shiftSignal(std::valarray<T> & data,
                std::valarray<T> & dataShifted,
                std::valarray<std::complex<T>> & spectrum,
                std::valarray<std::complex<T>> & phaseRamp,
                isce::signal::Signal<T> & sigObj);

        template<typename T>
        void shiftSignal(std::valarray<std::complex<T>> & data,
                std::valarray<std::complex<T>> & dataShifted,
                std::valarray<std::complex<T>> & spectrum,
                std::valarray<std::complex<T>> & phaseRamp,
                isce::signal::Signal<T> & sigObj);

        template<typename T>
        void frequencyResponseRange(size_t ncols, size_t nrows, 
                const double shift,
        	std::valarray<std::complex<T>> & shiftImpact);
        
        template<typename T>
        void frequencyResponseAzimuth(size_t ncols, size_t nrows, 
                const double shift,
                std::valarray<std::complex<T>> & shiftImpact);
         
    }
}

#endif

