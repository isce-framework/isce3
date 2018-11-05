//
// Author: Liang Yu
// Copyright 2018
//

#ifndef __ISCE_CUDA_CORE_GPUINTERPOLATOR_H__
#define __ISCE_CUDA_CORE_GPUINTERPOLATOR_H__

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#define CUDA_GLOBAL
#endif

#include "isce/core/Matrix.h"
#include <stdio.h>

using isce::core::Matrix;

// base interpolator is an abstract base class
namespace isce{ namespace cuda{ namespace core{
template <class U>
    class gpuInterpolator {
        public:
            CUDA_HOSTDEV gpuInterpolator(){};
            CUDA_DEV virtual U interpolate(double, double, const U*, size_t, size_t) = 0;
    };


// gpuBilinearInterpolator class derived from abstract gpuInterpolator class
template <class U>
class gpuBilinearInterpolator : public isce::cuda::core::gpuInterpolator<U> {
    public:
        CUDA_HOSTDEV gpuBilinearInterpolator(){};
        CUDA_DEV U interpolate(double, double, const U*, size_t, size_t);
        CUDA_HOST void interpolate_h(const Matrix<double>&, Matrix<U>&, double, double, U*);
};


// gpuBicubicInterpolator class derived from abstract gpuInterpolator class
template <class U>
class gpuBicubicInterpolator : public isce::cuda::core::gpuInterpolator<U> {
    public:
        CUDA_HOSTDEV gpuBicubicInterpolator(){};
        CUDA_DEV U interpolate(double, double, const U*, size_t, size_t);
        CUDA_HOST void interpolate_h(const Matrix<double>&, Matrix<U>&, double, double, U*);
};


// gpuSpline2dInterpolator class derived from abstract gpuInterpolator class
template <class U>
class gpuSpline2dInterpolator : public isce::cuda::core::gpuInterpolator<U> {
    protected:
        size_t _order;
    public:
        CUDA_HOSTDEV gpuSpline2dInterpolator(size_t order):_order(order){};
        CUDA_DEV U interpolate(double, double, const U*, size_t, size_t);
        CUDA_HOST void interpolate_h(const Matrix<double>&, Matrix<U>&, double, double, U*);
};


// gpuSinc2dInterpolator class derived from abstract gpuInterpolator class
template <class U>
class gpuSinc2dInterpolator : public isce::cuda::core::gpuInterpolator<U> {
    protected:
        double *kernel_d;
        int kernel_length, kernel_width;    // filter dimension idec=length, ilen=width
        int intpx, intpy;                   // sub-chip/image bounds? unclear...
        // True if initialized from host, false if copy-constructed from gpuSinc2dInterpolator on device
        bool owner;
    public:
        CUDA_HOSTDEV gpuSinc2dInterpolator(const gpuSinc2dInterpolator &i): 
            kernel_d(i.kernel_d), kernel_length(i.kernel_length), kernel_width(i.kernel_width), 
            intpx(i.intpx), intpy(i.intpy), owner(false) {};
        ~gpuSinc2dInterpolator();
        CUDA_HOST void sinc_coef(double, int, double, int);
        CUDA_DEV U interpolate(double, double, const U*, size_t, size_t);
        CUDA_HOST void interpolate_h(const Matrix<double>&, Matrix<U>&, double, double, U*);
};
}}}

#endif
