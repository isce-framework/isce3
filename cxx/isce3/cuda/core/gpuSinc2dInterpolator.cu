#include <cuda_runtime.h>
#include <iostream>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <valarray>
#include <vector>

#include <isce3/core/Matrix.h>
#include <isce3/cuda/except/Error.h>

#include "gpuInterpolator.h"

namespace isce3::cuda::core {

CUDA_HOST void compute_normalized_coefficients(
        const double beta, const int relfiltlen, const int decfactor,
        const double pedestal, thrust::host_vector<double>& filter)
{
    // Check if beta and pedestal within [0, 1]
    if (beta < 0 or beta > 1) {
        std::string err_str {"Beta value outside [0, 1]."};
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), err_str);
    }

    if (pedestal < 0 or pedestal > 1) {
        std::string err_str {"Pedestal value outside [0, 1]."};
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), err_str);
    }

    // Get filter size and check if it's compatible for normalization loop
    auto n_filter_coefs = static_cast<int>(filter.size());
    if (n_filter_coefs != decfactor * relfiltlen) {
        std::string err_str {"Filter size does not match expected number of coefficients."};
        throw isce3::except::LengthError(ISCE_SRCINFO(), err_str);
    }

    double wgthgt = (1.0 - pedestal) / 2.0;
    double soff = (n_filter_coefs - 1.) / 2.;

    // Compute filter coefficients
    std::vector<double> vec_filter(n_filter_coefs);
    for (int i = 0; i < n_filter_coefs; i++) {
        double weight = (1. - wgthgt) + (wgthgt * cos((M_PI * (i - soff)) / soff));
        double lag = (floor(i - soff) * beta) / (1. * decfactor);
        double sincResult = ((lag != 0.) ? (sin(M_PI * lag) / (M_PI * lag)) : 1.);
        // Assign product of sinc and weight to filter
        vec_filter[i] = sincResult * weight;
    }

    // Normalize filter coefficients and transpose values axes
    for (int i = 0; i < decfactor; ++i) {
        // Compute filter sum
        double ssum = 0.0;
        for (size_t j = 0; j < relfiltlen; ++j) {
            ssum += vec_filter[i + decfactor * j];
        }
        // Normalize the filter coefficients and copy to transposed kernel
        for (size_t j = 0; j < relfiltlen; ++j) {
            vec_filter[i + decfactor * j] /= ssum;
            filter[j + i * relfiltlen] = vec_filter[i + decfactor * j];
        }
    }
}

/* gpuSinc2dInterpolator helper functions below */
template<class T>
__global__ void gpuInterpolator_g(gpuSinc2dInterpolator<T> interp, double* x,
        double* y, const T* z, T* value, size_t nx, size_t ny = 0)
{
    /*
     *  GPU kernel to test interpolate() on the device for consistency.
        z := chip
     */
    int i = threadIdx.x;
    value[i] = interp.interpolate(x[i], y[i], z, nx, ny);
}

template<class T>
__host__ gpuSinc2dInterpolator<T>::gpuSinc2dInterpolator(
        const int kernelLength, const int decimationFactor, const double beta,
        const double pedestal) :
    _decimationFactor(decimationFactor),
    _kernelLength(kernelLength),
    _halfKernelLength(kernelLength / 2),
    _owner(true)
{
    // Temporary vector for storing sinc coefficients
    thrust::host_vector<double> h_filter(decimationFactor * kernelLength, 0.0);
    // beta = 1.0, pedestal = 0.0 below
    compute_normalized_coefficients(1.0, kernelLength, decimationFactor, 0.0,
            h_filter);

    // Malloc device-side memory (this API is host-side only)
    checkCudaErrors(cudaMalloc(&_kernel, h_filter.size() * sizeof(double)));

    // Copy kernel from host to device
    checkCudaErrors(cudaMemcpy(_kernel,
                thrust::raw_pointer_cast(h_filter.data()),
                h_filter.size() * sizeof(double), cudaMemcpyHostToDevice));
}

template<class T>
__device__ T gpuSinc2dInterpolator<T>::interpolate(
        double x, double y, const T* chip, size_t nx, size_t ny)
{
    /*
    definitions with respect to ResampSlc interpolate and sinc_eval_2d
    x   := fracAz
    y   := fracRg
    z   := chip
    nx  := chip length
    ny  := chip width
    */
    // Initialize return value
    T interp_val(0.0);

    // Separate interpolation coordinates into integer and fractional components
    const int ix = __double2int_rd(x);
    const int iy = __double2int_rd(y);
    const double frpx = x - ix;
    const double frpy = y - iy;

    // Check edge conditions
    bool x_in_bounds =
            ((ix >= (_halfKernelLength - 1)) && (ix <= (ny - _halfKernelLength - 1)));
    bool y_in_bounds =
            ((iy >= (_halfKernelLength - 1)) && (iy <= (nx - _halfKernelLength - 1)));
    if (x_in_bounds && y_in_bounds) {

        // Modify integer interpolation coordinates for sinc evaluation
        const int intpx = ix + _halfKernelLength;
        const int intpy = iy + _halfKernelLength;

        // Get nearest kernel indices
        // XXX Don't know how or why this even works!?
        int ifracx = min(max(0, int(frpx * _decimationFactor)), _decimationFactor - 1);
        int ifracy = min(max(0, int(frpy * _decimationFactor)), _decimationFactor - 1);

        // Compute weighted sum
        for (int i = 0; i < _kernelLength; i++) {
            for (int j = 0; j < _kernelLength; j++) {
                interp_val += chip[(intpy - i) * nx + intpx - j] *
                              T(_kernel[ifracy * _kernelLength + i]) *
                              T(_kernel[ifracx * _kernelLength + j]);
            }
        }
    }
    // Done
    return interp_val;
}

template<class T>
__host__ void gpuSinc2dInterpolator<T>::interpolate_h(
        const Matrix<double>& truth, Matrix<T>& chip, double start,
        double delta, T* h_z)
{
    /*
     *  CPU-side function to call the corresponding GPU function on a single
     thread for consistency checking truth = indices to interpolate to start,
     delta = unused h_z = output
     */

    // allocate host side memory
    size_t size_input_pts = truth.length() * sizeof(double);
    size_t size_output_pts = truth.length() * sizeof(T);
    double* h_x = (double*) malloc(size_input_pts);
    double* h_y = (double*) malloc(size_input_pts);

    // assign host side inputs
    for (size_t i = 0; i < truth.length(); ++i) {
        h_x[i] = truth(i, 0);
        h_y[i] = truth(i, 1);
    }

    size_t nx = chip.width();
    size_t ny = chip.length();

    // allocate device side memory
    double* d_x;
    checkCudaErrors(cudaMalloc((void**) &d_x, size_input_pts));
    double* d_y;
    checkCudaErrors(cudaMalloc((void**) &d_y, size_input_pts));
    T* d_z;
    checkCudaErrors(cudaMalloc((void**) &d_z, size_output_pts));
    T* d_chip;
    checkCudaErrors(cudaMalloc(
            (T**) &d_chip, chip.length() * chip.width() * sizeof(T)));

    // copy input data
    checkCudaErrors(
            cudaMemcpy(d_x, h_x, size_input_pts, cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(d_y, h_y, size_input_pts, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_chip, chip.data(),
            chip.length() * chip.width() * sizeof(T), cudaMemcpyHostToDevice));

    // launch!
    int n_threads = truth.length();
    gpuInterpolator_g<T>
            <<<1, n_threads>>>(*this, d_x, d_y, d_chip, d_z, nx, ny);

    // copy device results to host
    checkCudaErrors(
            cudaMemcpy(h_z, d_z, size_output_pts, cudaMemcpyDeviceToHost));

    // free memory
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(d_chip));
}

template<class T>
__host__ __device__ gpuSinc2dInterpolator<T>::~gpuSinc2dInterpolator()
{
#ifndef __CUDA_ARCH__
    if (_owner)
        checkCudaErrors(cudaFree(_kernel));
#endif
}

#define EXPLICIT_INSTANTIATION(T)                                               \
template class gpuSinc2dInterpolator<T>;                                       \
template __global__ void gpuInterpolator_g<T>(                                 \
        gpuSinc2dInterpolator<T> interp, double* x, double* y,                 \
        const T* z, T* value, size_t nx, size_t ny);

EXPLICIT_INSTANTIATION(float);
EXPLICIT_INSTANTIATION(thrust::complex<float>);
EXPLICIT_INSTANTIATION(double);
EXPLICIT_INSTANTIATION(thrust::complex<double>);
// XXX these template instantiations are needed in order to use
// `gpuSinc2dInterpolator` with `InterpolatorHandle`
EXPLICIT_INSTANTIATION(unsigned char);
EXPLICIT_INSTANTIATION(unsigned short);
EXPLICIT_INSTANTIATION(unsigned int);
}
