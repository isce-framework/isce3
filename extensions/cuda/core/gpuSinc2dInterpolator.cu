//
// Author: Liang Yu
// Copyright 2018
//

#include <iostream>
#include <stdio.h>
#include <valarray>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include "gpuInterpolator.h"
#include "../helper_cuda.h"

using isce::cuda::core::gpuInterpolator;
using isce::cuda::core::gpuSinc2dInterpolator;


/*
   each derived class needs it's own wrapper_d, gpuInterpolator_g...
*/

template <class U>
__host__ gpuSinc2dInterpolator<U>::gpuSinc2dInterpolator(int sincLen, int sincSub) :
        kernel_length(sincSub), kernel_width(sincLen), sinc_half(sincLen/2), 
        owner(true) {
    // Temporary valarray for storing sinc coefficients
    std::valarray<double> filter(0.0, sincSub * sincLen + 1);
    sinc_coef(1.0, sincLen, sincSub, 0.0, 1, filter);

    // Normalize filter
    for (size_t i = 0; i < sincSub; ++i) {
        // Compute filter sum
        double ssum = 0.0;
        for (size_t j = 0; j < sincLen; ++j) {
            ssum += filter[i + sincSub*j];
        }
        // Normalize the filter
        for (size_t j = 0; j < sincLen; ++j) {
            filter[i + sincSub*j] /= ssum;
        }
    }

    // Copy transpose of filter coefficients to member kernel matrix
    Matrix<double> h_kernel;
    h_kernel.resize(sincSub, sincLen);
    for (size_t i = 0; i < sincLen; ++i) {
        for (size_t j = 0; j < sincSub; ++j) {
            h_kernel(j,i) = filter[j + sincSub*i];
        }
    }

    // Malloc device-side memory (this API is host-side only)
    checkCudaErrors(cudaMalloc(&kernel, filter.size()*sizeof(double)));

    // Copy Orbit data to device-side memory and keep device pointer in gpuOrbit object. Device-side 
    // copy constructor simply shallow-copies the device pointers when called
    checkCudaErrors(cudaMemcpy(kernel, &(h_kernel[0]), filter.size()*sizeof(double), cudaMemcpyHostToDevice));
}


template <class U>
__global__ void gpuInterpolator_g(gpuSinc2dInterpolator<U> interp, double *x, double *y, const U *z, U *value, size_t nx, size_t ny=0) {
    /*
     *  GPU kernel to test interpolate() on the device for consistency.
        z := chip
     */
    int i = threadIdx.x;
    value[i] = interp.interpolate(x[i], y[i], z, nx, ny); 
}


template <class U>
__host__ void gpuSinc2dInterpolator<U>::interpolate_h(const Matrix<double>& truth, Matrix<U>& chip, double start, double delta, U* h_z) {
    /*
     *  CPU-side function to call the corresponding GPU function on a single thread for consistency checking
        truth = indices to interpolate to
        start, delta = unused
        h_z = output
     */

    // allocate host side memory
    size_t size_input_pts = truth.length() * sizeof(double);
    size_t size_output_pts = truth.length() * sizeof(U);
    double *h_x = (double *)malloc(size_input_pts);
    double *h_y = (double *)malloc(size_input_pts);

    // assign host side inputs
    for (size_t i = 0; i < truth.length(); ++i) {
        h_x[i] = truth(i,0);
        h_y[i] = truth(i,1);
    }

    size_t nx = chip.width();
    size_t ny = chip.length();

    // allocate device side memory
    double *d_x;
    checkCudaErrors(cudaMalloc((void**)&d_x, size_input_pts));
    double *d_y;
    checkCudaErrors(cudaMalloc((void**)&d_y, size_input_pts));
    U *d_z;
    checkCudaErrors(cudaMalloc((void**)&d_z, size_output_pts));
    U *d_chip;
    checkCudaErrors(cudaMalloc((U**)&d_chip, chip.length()*chip.width()*sizeof(U)));

    // copy input data
    checkCudaErrors(cudaMemcpy(d_x, h_x, size_input_pts, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, h_y, size_input_pts, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_chip, &chip.data()[0], chip.length()*chip.width()*sizeof(U), cudaMemcpyHostToDevice)); 

    // launch!
    int n_threads = truth.length();
    gpuInterpolator_g<U><<<1, n_threads>>>(*this, d_x, d_y, d_chip, d_z, nx, ny);
    
    // copy device results to host
    checkCudaErrors(cudaMemcpy(h_z, d_z, size_output_pts, cudaMemcpyDeviceToHost));

    // free memory
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(d_chip));
}


template <class U>
__host__ void 
gpuSinc2dInterpolator<U>::
        sinc_coef(double beta, double relfiltlen, int decfactor, double pedestal, int weight, std::valarray<double> & filter) { 

    int filtercoef = int(filter.size()) - 1;
    double wgthgt = (1.0 - pedestal) / 2.0;
    double soff = (filtercoef - 1.) / 2.;

    double wgt, s, fct;
    for (int i = 0; i < filtercoef; i++) {
        wgt = (1. - wgthgt) + (wgthgt * cos((M_PI * (i - soff)) / soff));
        s = (floor(i - soff) * beta) / (1. * decfactor);
        fct = ((s != 0.) ? (sin(M_PI * s) / (M_PI * s)) : 1.);
        filter[i] = ((weight == 1) ? (fct * wgt) : fct);
    }
}


template <class U>
__device__ U gpuSinc2dInterpolator<U>::interpolate(double x, double y, const U* chip, size_t nx, size_t ny) {
    /*
    definitions with respect to ResampSlc interpolate and sinc_eval_2d
    x   := fracAz
    y   := fracRg
    z   := chip
    nx  := chip length
    ny  := chip width
    */
    // Initialize return value
    U interp_val(0.0);

    // Separate interpolation coordinates into integer and fractional components
    const int ix = __double2int_rd(x);
    const int iy = __double2int_rd(y);
    const double frpx = x - ix;
    const double frpy = y - iy;

    // Check edge conditions
    bool x_in_bounds = ((ix >= (sinc_half - 1)) && (ix <= (ny - sinc_half - 1)));
    bool y_in_bounds = ((iy >= (sinc_half - 1)) && (iy <= (nx - sinc_half - 1)));
    if (x_in_bounds && y_in_bounds) {
    
        // Modify integer interpolation coordinates for sinc evaluation
        const int intpx = ix + sinc_half;
        const int intpy = iy + sinc_half;

        // Get nearest kernel indices
        int ifracx = min(max(0, int(frpx*kernel_length)), kernel_length-1);
        int ifracy = min(max(0, int(frpy*kernel_length)), kernel_length-1);

        // Compute weighted sum
        for (int i = 0; i < kernel_width; i++) {
            for (int j = 0; j < kernel_width; j++) {
                interp_val += chip[(intpy-i)*nx + intpx - j]
                     * U(kernel[ifracy*kernel_width + i])
                     * U(kernel[ifracx*kernel_width + j]);
            }
        }
    }
    // Done
    return interp_val;
}


template <class U>
__host__ __device__ gpuSinc2dInterpolator<U>::~gpuSinc2dInterpolator() {
#ifndef __CUDA_ARCH__
    if (owner)
        checkCudaErrors(cudaFree(kernel));
#endif
}

/*
 each template parameter needs it's own declaration here
 */
template class gpuSinc2dInterpolator<double>;
template class gpuSinc2dInterpolator<thrust::complex<double>>;
template class gpuSinc2dInterpolator<thrust::complex<float>>;

template __global__ void
gpuInterpolator_g<double>(gpuSinc2dInterpolator<double> interp, double *x, double *y,
                                  const double *z, double *value, size_t nx, size_t ny);
template __global__ void
gpuInterpolator_g<thrust::complex<double>>(gpuSinc2dInterpolator<thrust::complex<double>> interp, double *x, double *y,
                                  const thrust::complex<double> *z, thrust::complex<double> *value, size_t nx, size_t ny);
template __global__ void
gpuInterpolator_g<thrust::complex<float>>(gpuSinc2dInterpolator<thrust::complex<float>> interp, double *x, double *y,
                                  const thrust::complex<float> *z, thrust::complex<float> *value, size_t nx, size_t ny);
