#pragma once

#include <isce3/core/forward.h>
#include <isce3/core/Common.h>

#include <thrust/host_vector.h>

using isce3::core::Matrix;

/** base interpolator is an abstract base class */
namespace isce3{ namespace cuda{ namespace core{
template <class U>
    class gpuInterpolator {
        public:
            CUDA_HOSTDEV gpuInterpolator() {}
            CUDA_HOSTDEV virtual ~gpuInterpolator() {}
            CUDA_DEV virtual U interpolate(double, double, const U*, size_t, size_t) = 0;
    };


/** gpuBilinearInterpolator class derived from abstract gpuInterpolator class */
template <class U>
class gpuBilinearInterpolator : public isce3::cuda::core::gpuInterpolator<U> {
    public:
        CUDA_HOSTDEV gpuBilinearInterpolator(){};
        CUDA_DEV U interpolate(double, double, const U*, size_t, size_t);
        CUDA_HOST void interpolate_h(const Matrix<double>&, Matrix<U>&, double, double, U*);
};


/** gpuBicubicInterpolator class derived from abstract gpuInterpolator class */
template <class U>
class gpuBicubicInterpolator : public isce3::cuda::core::gpuInterpolator<U> {
    public:
        CUDA_HOSTDEV gpuBicubicInterpolator(){};
        CUDA_DEV U interpolate(double, double, const U*, size_t, size_t);
        CUDA_HOST void interpolate_h(const Matrix<double>&, Matrix<U>&, double, double, U*);
};


/** gpuSpline2dInterpolator class derived from abstract gpuInterpolator class */
template <class U>
class gpuSpline2dInterpolator : public isce3::cuda::core::gpuInterpolator<U> {
    protected:
        size_t _order;
    public:
        CUDA_HOSTDEV gpuSpline2dInterpolator(size_t order):_order(order){};
        CUDA_DEV U interpolate(double, double, const U*, size_t, size_t);
        CUDA_HOST void interpolate_h(const Matrix<double>&, Matrix<U>&, double, double, U*);
};


/** gpuSinc2dInterpolator class derived from abstract gpuInterpolator class */
template <class U>
class gpuSinc2dInterpolator : public isce3::cuda::core::gpuInterpolator<U> {
    protected:
        double *_kernel;
        // Number of divisions per sample (total number samples in the lookup
        // table is kernel length * decimation factor)
        int _decimationFactor;
        // size of kernel
        int _kernelLength;
        // size of half kernel
        int _halfKernelLength;
        // True if initialized from host,
        // False if copy-constructed from gpuSinc2dInterpolator on device
        bool _owner;
    public:
        CUDA_HOSTDEV gpuSinc2dInterpolator(){};

        /** Host constructor where filter construction is performed on host.
         *
         * \param[in] kernelLength      size of kernel
         * \param[in] decimationFactor  number of divisions per sample (total
         *                              number samples in the lookup table is
         *                              kernel length * decimation factor)
         * \param[in] beta              bandwidth of the filter [0, 1]
         * \param[in] pedestal          window parameter [0, 1]
         */
        CUDA_HOST gpuSinc2dInterpolator(
                const int kernelLength, const int decimationFactor,
                const double beta = 1.0, const double pedestal = 0.0);

        /** Device constructor where filter construction is performed on host
         *  and the result is passed to constructor. _owner set to false as
         *  device_filter is initialized, populated, and persisted outside this
         *  constructor context.
         *
         * \param[in] device_filter     Pointer filter data on device
         * \param[in] kernelLength      size of kernel
         * \param[in] decimationFactor  number of divisions per sample (total
         *                              number samples in the lookup table is
         *                              kernel length * decimation factor)
         */
        CUDA_DEV gpuSinc2dInterpolator(double *device_filter, int kernelLength,
                int decimationFactor) :
            _kernel(device_filter),
            _decimationFactor(decimationFactor),
            _kernelLength(kernelLength),
            _halfKernelLength(kernelLength / 2),
            _owner(false) {};

        CUDA_HOSTDEV gpuSinc2dInterpolator(const gpuSinc2dInterpolator &i):
            _kernel(i._kernel),
            _decimationFactor(i._decimationFactor),
            _kernelLength(i._kernelLength),
            _halfKernelLength(i._halfKernelLength),
            _owner(false) {};

        CUDA_HOSTDEV ~gpuSinc2dInterpolator();
        CUDA_DEV U interpolate(double, double, const U*, size_t, size_t);
        CUDA_HOST void interpolate_h(const Matrix<double>&, Matrix<U>&, double, double, U*);
};

/** Construct a filter for sinc interpolator windowed with a cosine.
 *
 * \param[in] beta          bandwidth parameter [0, 1]
 * \param[in] relfiltlen    total number of filter taps
 * \param[in] decfactor     decimation factor - number of divisions between
 *                          samples
 * \param[in] pedestal      window parameter [0, 1]
 * \param[out]  filter      Filter whose values are to be computed
 */
CUDA_HOST void compute_normalized_coefficients(
        const double beta, const int relfiltlen, const int decfactor,
        const double pedestal, thrust::host_vector<double>& filter);


/** gpuNearestNeighborInterpolator class derived from abstract gpuInterpolator class */
template <class T>
class gpuNearestNeighborInterpolator : public isce3::cuda::core::gpuInterpolator<T> {
    public:
        CUDA_HOSTDEV gpuNearestNeighborInterpolator(){};
        CUDA_DEV T interpolate(double x, double y, const T* z, size_t nx, size_t ny = 0);
};

}}}
