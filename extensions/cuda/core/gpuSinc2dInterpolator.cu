//
// Author: Liang Yu
// Copyright 2018
//

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "gpuInterpolator.h"
#include "gpuComplex.h"
#include "../helper_cuda.h"

using isce::cuda::core::gpuInterpolator;
using isce::cuda::core::gpuSinc2dInterpolator;
using isce::cuda::core::gpuComplex;


/*
   each derived class needs it's own wrapper_d, gpuInterpolator_g...
*/

template <class U>
__device__ void wrapper_d(gpuSinc2dInterpolator<U> interp, double x, double y, const U *z, U *value, size_t nx, size_t ny=0) {
    /*
     *  device side wrapper used to get map interfaces of actual device function to global test function
     */
    *value = interp.interpolate(x, y, z, nx, ny); 
}


template <class U>
__global__ void gpuInterpolator_g(gpuSinc2dInterpolator<U> interp, double *x, double *y, const U *z, U *value, size_t nx, size_t ny=0) {
    /*
     *  GPU kernel to test interpolate() on the device for consistency.
     */
    int i = threadIdx.x;
    wrapper_d(interp, x[i], y[i], z, &value[i], nx, ny);
}


template <class U>
__host__ void gpuSinc2dInterpolator<U>::interpolate_h(const Matrix<double>& truth, Matrix<U>& m, double start, double delta, U* h_z) {
    /*
     *  CPU-side function to call the corresponding GPU function on a single thread for consistency checking
     */

    // allocate host side memory
    size_t size_input_pts = truth.length() * sizeof(double);
    size_t size_output_pts = truth.length() * sizeof(U);
    double *h_x = (double *)malloc(size_input_pts);
    double *h_y = (double *)malloc(size_input_pts);

    // assign host side inputs
    for (size_t i = 0; i < truth.length(); ++i) {
        h_x[i] = (truth(i,0) - start) / delta;
        h_y[i] = (truth(i,1) - start) / delta;
    }

    size_t nx = m.width();
    size_t ny = m.length();

    // allocate devie side memory
    double *d_x;
    checkCudaErrors(cudaMalloc((void**)&d_x, size_input_pts));
    double *d_y;
    checkCudaErrors(cudaMalloc((void**)&d_y, size_input_pts));
    U *d_z;
    checkCudaErrors(cudaMalloc((void**)&d_z, size_output_pts));
    U *d_m;
    checkCudaErrors(cudaMalloc((U**)&d_m, m.length()*m.width()*sizeof(U)));

    // copy input data
    checkCudaErrors(cudaMemcpy(d_x, h_x, size_input_pts, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, h_y, size_input_pts, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_m, &m.data()[0], m.length()*m.width()*sizeof(U), cudaMemcpyHostToDevice)); 

    // launch!
    int n_threads = truth.length();
    gpuInterpolator_g<U><<<1, n_threads>>>(*this, d_x, d_y, d_m, d_z, nx, ny);
    
    // copy device results to host
    checkCudaErrors(cudaMemcpy(h_z, d_z, size_output_pts, cudaMemcpyDeviceToHost));

    // free memory
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(d_m));
}


template <class U>
__host__ void gpuSinc2dInterpolator<U>::sinc_coef(double beta, int decfactor, double pedestal, int weight) { 
    size_t n_coefs = kernel_length*kernel_width - 1;
    std::vector<double> kernel_h(n_coefs);
    size_t kernel_sz = n_coefs * sizeof(double);
    double wgthgt = (1.0 - pedestal) / 2.0;
    double soff = (n_coefs - 1.) / 2.;

    double wgt, s, fct;
    for (int i = 0; i < n_coefs; i++) {
        wgt = (1. - wgthgt) + (wgthgt * cos((M_PI * (i - soff)) / soff));
        s = (floor(i - soff) * beta) / (1. * decfactor);
        fct = ((s != 0.) ? (sin(M_PI * s) / (M_PI * s)) : 1.);
        kernel_h[i] = ((weight == 1) ? (fct * wgt) : fct);
    }

    checkCudaErrors(cudaMalloc((void**)&kernel_d, kernel_sz));
    checkCudaErrors(cudaMemcpy(kernel_d, kernel_h.data(), kernel_sz, cudaMemcpyHostToDevice));
    owner = true;
}


template <class U>
__device__ U gpuSinc2dInterpolator<U>::interpolate(double x, double y, const U* z, size_t nx, size_t ny) {
    /*
    definitions with respect to ResampSlc
    x   := fracAz
    y   := fracRg
    z   := chip
    nx  := chip length
    ny  := chip width
    */
    // Initialize return value
    U ret(0.0);
    // Interpolate for valid indices
    if ((intpx >= (kernel_width-1)) && (intpx < nx) && (intpy >= (kernel_width-1)) && (intpy < ny)) {
        // Get nearest kernel indices
        int ifracx = min(max(0, int(x)*kernel_length), kernel_length-1);
        int ifracy = min(max(0, int(y)*kernel_length), kernel_length-1);
        // Compute weighted sum
        // return _data[row*_width + column];
        for (int i = 0; i < kernel_width; i++) {
            for (int j = 0; j < kernel_width; j++) {
                ret += z[(intpx-i)*nx + intpy - j]
                     * kernel_d[ifracx*kernel_width + i]
                     * kernel_d[ifracy*kernel_width + j];
            }
        }
    }
    // Done
    return ret;
}


template <class U>
gpuSinc2dInterpolator<U>::~gpuSinc2dInterpolator() {
    if (owner)
        checkCudaErrors(cudaFree(kernel_d));
}

/*
 each template parameter needs it's own declaration here
 */
template class gpuSinc2dInterpolator<double>;
template class gpuSinc2dInterpolator<gpuComplex<double>>;

template __global__ void
gpuInterpolator_g<double>(gpuSinc2dInterpolator<double> interp, double *x, double *y,
                                  const double *z, double *value, size_t nx, size_t ny);

template __global__ void
gpuInterpolator_g<gpuComplex<double>>(gpuSinc2dInterpolator<gpuComplex<double>> interp, double *x, double *y,
                                  const double *z, double *value, size_t nx, size_t ny);
