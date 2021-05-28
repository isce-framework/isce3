#include "InterpolatorHandle.h"

#include <isce3/cuda/except/Error.h>
#include <isce3/except/Error.h>

#include <pyre/journal.h>

#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

template<class T>
using DeviceInterp = isce3::cuda::core::gpuInterpolator<T>;

namespace isce3::cuda::core {

template<class T>
__global__ void init_interp(
        DeviceInterp<T>** interp, isce3::core::dataInterpMethod interp_method,
        bool * unsupported_interp)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Choose interpolator
        switch(interp_method) {
            case isce3::core::BILINEAR_METHOD:
                (*interp) = new isce3::cuda::core::gpuBilinearInterpolator<T>();
                break;
            case isce3::core::BICUBIC_METHOD:
                (*interp) = new isce3::cuda::core::gpuBicubicInterpolator<T>();
                break;
            case isce3::core::BIQUINTIC_METHOD:
                {
                size_t order = 6;
                (*interp) = new isce3::cuda::core::gpuSpline2dInterpolator<T>(order);
                break;
                }
            default:
                *unsupported_interp = true;
                break;
        }
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

    thrust::device_vector<bool> d_unsupported_interp(1, false);
    init_interp<<<1, 1>>>(_interp, interp_method, d_unsupported_interp.data().get());
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    bool unsupported_interp = d_unsupported_interp[0];
    if (unsupported_interp)
    {
        pyre::journal::error_t error(
                "isce.cuda.core.InterpolatorHandle.InterpolatorHandle");
        error << "Unsupported interpolator method provided."
              << pyre::journal::endl;
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                                         "Unsupported interpolator method provided.");
    }
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
} // namespace isce3::cuda::core
