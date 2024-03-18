#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include <pyre/journal.h>

#include <isce3/core/Constants.h>
#include <isce3/cuda/except/Error.h>
#include <isce3/except/Error.h>

#include "InterpolatorHandle.h"

using isce3::core::SINC_LEN;
using isce3::core::SINC_SUB;
template<class T>
using DeviceInterp = isce3::cuda::core::gpuInterpolator<T>;

namespace isce3::cuda::core {

/* Function that initializes an interpolator for interpolator handle
 *
 * \param[out]  interp      Pointer to gpuInterpolator on device
 * \param[in] interp_method Enum that determines the type of interpolator to
 *                          initialize
 * \param[in] filter        Array of tablulated filter coefficients to
 *                          initialize sinc interpolator with
 */
template<class T>
__global__ void init_interp(DeviceInterp<T>** interp,
        isce3::core::dataInterpMethod interp_method, double* filter)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Choose interpolator
        if (interp_method == isce3::core::SINC_METHOD) {
            (*interp) = new
                isce3::cuda::core::gpuSinc2dInterpolator<T>(filter,
                        SINC_LEN, SINC_SUB);
        }

        else if (interp_method == isce3::core::BILINEAR_METHOD)
            (*interp) = new isce3::cuda::core::gpuBilinearInterpolator<T>();

        else if (interp_method == isce3::core::BICUBIC_METHOD)
            (*interp) = new isce3::cuda::core::gpuBicubicInterpolator<T>();

        else if (interp_method == isce3::core::BIQUINTIC_METHOD) {
            size_t order = 6;
            (*interp) = new isce3::cuda::core::gpuSpline2dInterpolator<T>(order);
        }

        else if (interp_method == isce3::core::NEAREST_METHOD)
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
            && interp_method != isce3::core::NEAREST_METHOD
            && interp_method != isce3::core::SINC_METHOD)
    {
        pyre::journal::error_t error(
                "isce3.cuda.core.InterpolatorHandle.InterpolatorHandle");
        error << "Unsupported interpolator method provided."
              << pyre::journal::endl;
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Unsupported interpolator method provided.");
    }

    // Pointer to sinc interpolator filter. Defaults to nullptr except for sinc
    // interpolator. If sinc interpolator, then compute filter and assign
    // pointer to filter.
    double* filter_ptr = nullptr;
    if (interp_method == isce3::core::SINC_METHOD) {
        // Temporary host_vector storing normalized sinc filter coefficients
        thrust::host_vector<double> h_filter(SINC_SUB * SINC_LEN, 0.0);
        // beta = 1.0, pedestal = 0.0 below
        compute_normalized_coefficients(
                1.0, SINC_LEN, SINC_SUB, 0.0, h_filter);

        // Copy to device_vector class member to persist beyond scope of
        // constructor
        d_sinc_filter = h_filter;

        // Set sinc interpolator filter pointer to device_vector class member
        filter_ptr = thrust::raw_pointer_cast(d_sinc_filter.data());
    }

    // This is basically reproducing the functionality of the (host)
    // gpuSinc2dInterpolator constructor. It's copy-pasted here because we must
    // create the actual interpolator object *on the device* so that we may
    //call its virtual functions from the device.
    // For more detail:
    // https://github-fn.jpl.nasa.gov/isce-3/isce/pull/1399#discussion_r17565
    init_interp<<<1, 1>>>(_interp, interp_method, filter_ptr);
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
template class InterpolatorHandle<unsigned short>;
template class InterpolatorHandle<unsigned int>;
} // namespace isce3::cuda::core
