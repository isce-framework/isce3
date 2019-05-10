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
        template <typename T> T sinc(T t);
        // The kernel classes
        template <typename T> class Kernel;
        template <typename T> class BartlettKernel;
        template <typename T> class LinearKernel;
        template <typename T> class KnabKernel;
    }
}

/** Abstract base class for all kernels.
 *
 * Basically just a closure around an arbitrary function and a width property.
 * Template parameter T defines type for coefficients, typically float or
 * double.
 */
template <typename T>
class isce::core::Kernel {

    public:
        /** Virtual destructor (allow destruction of base Kernel pointer) */
        virtual ~Kernel() {}

        /** Evaluate kernel at given location in [-halfwidth, halfwidth] */
        virtual T operator()(double x) const = 0;

        /** Get width of kernel.
         *
         * Units are the same as are used for calls to operator().
         */
        double width() const {return _halfwidth*2;}

    protected:
        double _halfwidth;
};

/** Bartlett kernel (triangle function). */
template <typename T>
class isce::core::BartlettKernel : public isce::core::Kernel<T> {

    public:
        /** Triangle function constructor. */
        BartlettKernel(double width);
        T operator()(double x) const override;
};

/** Linear kernel, which is just a special case of Bartlett. */
template <typename T>
class isce::core::LinearKernel : public isce::core::BartlettKernel<T> {

    public:
        LinearKernel() : isce::core::BartlettKernel<T>(2.0) {}
};


/** Kernel based on the paper by Knab for interpolating band-limited signals.
 *
 * For details see references
 * @cite knab1983
 * @cite migliaccio2007
 */
template <typename T>
class isce::core::KnabKernel : public isce::core::Kernel<T> {

    public:
        /** Constructor of Knab's kernel.
         *
         * @param[in] width     Total width of kernel.
         * @param[in] bandwidth Bandwidth of signal to be interpolated, as a
         *                      fraction of the sample rate (0 < bandwidth < 1).
         */
        KnabKernel(double width, double bandwidth);

        T operator()(double x) const override;

        /** Get bandwidth of kernel. */
        double bandwidth() const {return _bandwidth;}

    private:
        double _bandwidth;
};

/** sinc function defined as \f$ \frac{\sin(\pi x)}{\pi x} \f$ */
template <typename T>
T
isce::core::sinc(T t);

#endif
