#ifndef ISCE_LIB_SHIFTSIGNAL_H
#define ISCE_LIB_SHIFTSIGNAL_H

#include <cmath>
#include <valarray>
#include "Signal.h"
#include "Filter.h"

namespace isce {
    namespace signal {

        template<typename T, typename U>
        void shiftSignal(std::valarray<T> & data,
            std::valarray<T> & dataShifted,
            std::valarray<std::complex<U>> & spectrum,
            size_t ncols, size_t nrows,
            const double shiftX, const double shiftY,
            isce::signal::Signal<U> & sigObj);

        template<typename T, typename U>
        void shiftSignal(std::valarray<T> & data,
            std::valarray<T> & dataShifted,
            std::valarray<std::complex<U>> & spectrum,
            std::valarray<std::complex<U>> & phaseRamp,
            isce::signal::Signal<U> & sigObj);

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
