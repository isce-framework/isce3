//
// Author: Brian Hawkins
// Copyright 2019
//

#ifndef ISCE_CORE_INTERP1D_H
#define ISCE_CORE_INTERP1D_H

#include <valarray>

// isce::core
#include "Constants.h"
#include "Kernels.h"

// Declaration
namespace isce {
    namespace core {
        template <typename TK> class Interp1d;
    }
}

/** Class for 1D interpolation of uniformly sampled data. */
template <typename TK>
class isce::core::Interp1d {

    // Public interface
    public:
        /** Generic 1D interpolation constructor.
         *
         * @param[in] kernel Arbitrary kernel function.
         */
        Interp1d(const std::shared_ptr<isce::core::Kernel<TK>> &kernel) : _kernel(kernel) {
            // XXX Throw error if kernel->width() is not an integer?
            _width = (unsigned) ceil(kernel->width());
        };

        // Factories
        /** Linear 1D interpolator. */
        static Interp1d<TK> Linear();

        /** 1D interpolation with Knab pulse.
         *
         * @param[in] width     Total width (in samples) of interpolator.
         * @param[in] bandwidth Bandwidth of signal to be interpolated, as a
         *                      fraction of the sample rate (0 < bandwidth < 1).
         */
        static Interp1d<TK> Knab(double width, double bandwidth);

        /** Interpolate sequence x at point t
         *
         * @param[in] x Sequence to interpolate.
         * @param[in] t Desired time sample (0 <= t <= x.size()-1).
         * @returns Interpolated value or 0 if kernel would run off array.
         */
        template<typename TD>
        TD interp(const std::valarray<TD> &x, double t);

        /** Get width of kernel. */
        unsigned width() {return _width;}

    // Protected constructor and data to be used by derived classes
    protected:
        unsigned _width;
        std::shared_ptr<isce::core::Kernel<TK>> _kernel;
};

// Get inline implementations
#define ISCE_CORE_INTERP1D_ICC
#include "Interp1d.icc"
#undef ISCE_CORE_INTERP1D_ICC

#endif
