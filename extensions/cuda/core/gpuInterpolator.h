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
}}}

/*
// Akima class derived from abstract gpuInterpolator class
template <class U>
class isce::cuda::core::Akima {
    public:
        U interpolate(double, double, const Matrix<U> &);
};


*/

#endif
