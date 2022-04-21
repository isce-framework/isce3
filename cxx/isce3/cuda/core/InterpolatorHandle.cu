#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include <pyre/journal.h>

#include <isce3/cuda/except/Error.h>
#include <isce3/except/Error.h>

#include "InterpolatorHandle.h"

template<class T>
using DeviceInterp = isce3::cuda::core::gpuInterpolator<T>;

namespace isce3::cuda::core {

template<class T>
__global__ void init_interp(DeviceInterp<T>** interp,
        isce3::core::dataInterpMethod interp_method)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Choose interpolator
        if (interp_method == isce3::core::BILINEAR_METHOD)
            (*interp) = new isce3::cuda::core::gpuBilinearInterpolator<T>();

        if (interp_method == isce3::core::BICUBIC_METHOD)
            (*interp) = new isce3::cuda::core::gpuBicubicInterpolator<T>();

        if (interp_method == isce3::core::BIQUINTIC_METHOD) {
            size_t order = 6;
            (*interp) = new isce3::cuda::core::gpuSpline2dInterpolator<T>(order);
        }

        if (interp_method == isce3::core::NEAREST_METHOD)
            (*interp) = new isce3::cuda::core::gpuNearestNeighborInterpolator<T>();
    }
}

template<class T>
__global__ void finalize_interp(DeviceInterp<T>** interp)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *interp;
    }
}

template<class T>
InterpolatorHandle<T>::InterpolatorHandle(
        isce3::core::dataInterpMethod interp_method)
{
    checkCudaErrors(cudaMalloc(&_interp, sizeof(DeviceInterp<T>**)));

    if (interp_method != isce3::core::BILINEAR_METHOD
            && interp_method != isce3::core::BICUBIC_METHOD
            && interp_method != isce3::core::BIQUINTIC_METHOD
            && interp_method != isce3::core::NEAREST_METHOD)
    {
        pyre::journal::error_t error(
                "isce3.cuda.core.InterpolatorHandle.InterpolatorHandle");
        error << "Unsupported interpolator method provided."
              << pyre::journal::endl;
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Unsupported interpolator method provided.");
    }

    thrust::device_vector<bool> d_unsupported_interp(1, false);
    init_interp<<<1, 1>>>(_interp, interp_method);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

template<class T>
InterpolatorHandle<T>::~InterpolatorHandle()
{
    finalize_interp<<<1, 1>>>(_interp);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(_interp));
}

template class InterpolatorHandle<float>;
template class InterpolatorHandle<thrust::complex<float>>;
template class InterpolatorHandle<double>;
template class InterpolatorHandle<thrust::complex<double>>;
template class InterpolatorHandle<unsigned char>;
template class InterpolatorHandle<unsigned int>;
} // namespace isce3::cuda::core
