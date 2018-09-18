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
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#endif


// base interpolator is an abstract base class
namespace isce{ namespace cuda{ namespace core{
template <class U>
    class gpuInterpolator {
        public:
            CUDA_DEV virtual U interpolate(void) = 0;
    };
}}}


// gpuBilinearInterpolator class derived from abstract gpuInterpolator class
namespace isce{ namespace cuda{ namespace core{
template <class U>
class gpuBilinearInterpolator : public isce::cuda::core::gpuInterpolator<U> {
    public:
        CUDA_DEV U interpolate(double, double, const U*, size_t);
    };
}}}

/*
// Bicubic class derived from abstract gpuInterpolator class
template <class U>
class isce::cuda::core::Bicubic {
    public:
        U interpolate(double, double, const Matrix<U> &);
};


// Akima class derived from abstract gpuInterpolator class
template <class U>
class isce::cuda::core::Akima {
    public:
        U interpolate(double, double, const Matrix<U> &);
};


// Bilinear class derived from abstract gpuInterpolator class
template <class U>
class isce::cuda::core::Bilinear {
    public:
        U interpolate(double, double, const Matrix<U> &);
};
*/

#endif
