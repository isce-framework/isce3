//
// Author: Liang Yu
// Copyright 2018
//

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "gpuInterpolator.h"
#include "../helper_cuda.h"

using isce::cuda::core::gpuInterpolator;
using isce::cuda::core::gpuBilinearInterpolator;


template <class U>
__global__ void gpuInterpolator_g(gpuBilinearInterpolator<U> interp, double *x, double *y, const U *z, U *value, size_t nx, size_t ny=0) {
    /*
     *  GPU kernel to test interpolate() on the device for consistency.
     */
    int i = threadIdx.x;
    value[i] = interp.interpolate(x[i], y[i], z, nx, ny); 
}


template <class U>
__host__ void isce::cuda::core::gpuBilinearInterpolator<U>::interpolate_h(const Matrix<double>& truth, Matrix<U>& m, double start, double delta, U* h_z) {
    /*
     *  CPU-side function to call the corresponding GPU function on a single thread for consistency checking
     */

    // allocate host side memory
    size_t size_input_pts = truth.length() * sizeof(double);
    size_t size_output_pts = truth.length() * sizeof(U);
    size_t nx = m.width();

    double *h_x = (double *)malloc(size_input_pts);
    double *h_y = (double *)malloc(size_input_pts);

    // assign host side inputs
    for (size_t i = 0; i < truth.length(); ++i) {
        h_x[i] = (truth(i,0) - start) / delta;
        h_y[i] = (truth(i,1) - start) / delta;
    }

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
    gpuInterpolator_g<U><<<1, n_threads>>>(*this, d_x, d_y, d_m, d_z, nx);
    
    // copy device results to host
    checkCudaErrors(cudaMemcpy(h_z, d_z, size_output_pts, cudaMemcpyDeviceToHost));

    // free memory
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(d_m));
}


template <class U>
__device__ U isce::cuda::core::gpuBilinearInterpolator<U>::interpolate(double x, double y, const U* z, size_t nx, size_t ny=0) {
    size_t x1 = floor(x);
    size_t x2 = ceil(x);
    size_t y1 = floor(y);
    size_t y2 = ceil(y);
    
    U q11 = z[y1*nx + x1];
    U q12 = z[y2*nx + x1];
    U q21 = z[y1*nx + x2];
    U q22 = z[y2*nx + x2];

    if ((y1 == y2) && (x1 == x2)) {
        return q11;
    } else if (y1 == y2) {
        return ((U)((x2 - x) / (x2 - x1)) * q11) +
               ((U)((x - x1) / (x2 - x1)) * q21);
    } else if (x1 == x2) {
        return ((U)((y2 - y) / (y2 - y1)) * q11) +
               ((U)((y - y1) / (y2 - y1)) * q12);
    } else {
        return  ((q11 * (U)((x2 - x) * (y2 - y))) /
                 (U)((x2 - x1) * (y2 - y1))) +
                ((q21 * (U)((x - x1) * (y2 - y))) /
                 (U)((x2 - x1) * (y2 - y1))) +
                ((q12 * (U)((x2 - x) * (y - y1))) /
                 (U)((x2 - x1) * (y2 - y1))) +
                ((q22 * (U)((x - x1) * (y - y1))) /
                 (U)((x2 - x1) * (y2 - y1)));
    }
}


/*
 each template parameter needs it's own declaration here
 */
template class gpuBilinearInterpolator<double>;
template class gpuBilinearInterpolator<float>;

template __global__ void
gpuInterpolator_g<double>(gpuBilinearInterpolator<double> interp, double *x, double *y,
                          const double *z, double *value, size_t nx, size_t ny);

