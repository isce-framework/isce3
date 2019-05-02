//
// Author: Brian Hawkins
// Copyright 2019
//

#ifndef ISCE_CORE_KERNELS_H
#define ISCE_CORE_KERNELS_H

#include <valarray>

// isce::core
#include "Constants.h"

// Declaration
namespace isce {
    namespace core {
        // The kernel classes
        template <typename T> class Kernel;
        template <typename T> class BartlettKernel;
        template <typename T> class KnabKernel;
    }
}

// Definition of abstract base class for all kernels.  Basically just a closure
// around an arbitrary function and a width property.
// Parameter T defines tupe for coefficients.  Typically float or double.
template <typename T>
class isce::core::Kernel {

    // Public interface
    public:
        /** Virtual destructor (allow destruction of base Kernel pointer) */
        virtual ~Kernel() {}

        /** Evaluate kernel at given location in [-halfwidth, halfwidth] */
        virtual T operator()(double x) = 0;

        /** Get width of kernel. */
        double width() {return _halfwidth*2;}

    // Protected constructor and data to be used by derived classes
    protected:
        double _halfwidth;
};

template <typename T>
class isce::core::BartlettKernel : public isce::core::Kernel<T> {

    public:
        /** Linear kernel, aka Bartlett or triangle function. */
        BartlettKernel(double width);
        T operator()(double x);
};

template <typename T>
class isce::core::KnabKernel : public isce::core::Kernel<T> {

    public:
        /** Constructor for kernel corresponding to paper by Knab, 1983.
         *
         * @param[in] width     Total width (in samples) of kernel.
         * @param[in] bandwidth Bandwidth of signal to be interpolated, as a
         *                      fraction of the sample rate (0 < bandwidth < 1).
         */
        KnabKernel(double width, double bandwidth);

        T operator()(double x);

        /** Get bandwidth of kernel. */
        double bandwidth() {return _bandwidth;}

    private:
        double _bandwidth;
};

#endif
