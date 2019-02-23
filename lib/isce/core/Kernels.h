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

// Definition of parent Kernel
template <typename T>
class isce::core::Kernel {

    // Public interface
    public:
        /** Virtual destructor (allow destruction of base Kernel pointer) */
        virtual ~Kernel() {}

        /** Evaluate kernel at given location in [-hw, hw] */
        virtual T operator()(double x) = 0;
        
        /** Get width of kernel. */
        double width() {return _halfwidth*2;}

    // Protected constructor and data to be used by derived classes
    protected:
        double _halfwidth;
};

// Linear kernel, aka Bartlett
template <typename T>
class isce::core::BartlettKernel : public isce::core::Kernel<T> {

    public:
        BartlettKernel(double width); 
        T operator()(double x);
};

// sampling window from Knab 1983
template <typename T>
class isce::core::KnabKernel : public isce::core::Kernel<T> {
    
    public:
        KnabKernel(double width, double bandwidth);
        T operator()(double x);

        /** Get bandwidth of kernel. */
        double bandwidth() {return _bandwidth;}

    private:
        double _bandwidth;
};

#endif
