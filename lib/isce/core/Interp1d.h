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
        // The kernel classes
        template <typename TD, typename TK> class Interp1d;
    }
}

template <typename TD, typename TK>
class isce::core::Interp1d {

    // Public interface
    public:
        /** Constructors */
        // Interp1d(isce::core::Kernel<TK> *kernel) : _kernel(kernel) {
        Interp1d(const std::shared_ptr<isce::core::Kernel<TK>>& kernel) : _kernel(kernel) {
            // XXX Throw error if kernel->width() is an integer?
            _width = (unsigned) ceil(kernel->width());
        };
        // Factories
        static Interp1d Linear();
        static Interp1d Knab(double width, double bandwidth);

        /** Interpolate sequence x at point t  */
        TD interp(std::valarray<TD> &x, double t);
        
        /** Get width of kernel. */
        unsigned width() {return _width;}

    // Protected constructor and data to be used by derived classes
    protected:
        unsigned _width;
        std::shared_ptr<isce::core::Kernel<TK>> _kernel;
};

#endif
