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
using isce::cuda::core::gpuBicubicInterpolator;


/*
   each derived class needs it's own wrapper_d, gpuInterpolator_g...
*/
template <class U>
__device__ void wrapper_d(gpuBicubicInterpolator<U> interp, double x, double y, const U *z, U *value, size_t nx, size_t ny=0) {
    /*
     *  device side wrapper used to get map interfaces of actual device function to global test function
     */
    *value = interp.interpolate(x, y, z, nx); 
}


template <class U>
__global__ void gpuInterpolator_g(gpuBicubicInterpolator<U> interp, double *x, double *y, const U *z, U *value, size_t nx, size_t ny=0) {
    /*
     *  GPU kernel to test interpolate() on the device for consistency.
     */
    int i = threadIdx.x;
    wrapper_d(interp, x[i], y[i], z, &value[i], nx);
}


template <class U>
__host__ void isce::cuda::core::gpuBicubicInterpolator<U>::interpolate_h(const Matrix<double>& truth, Matrix<U>& m, double start, double delta, U* h_z) {
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
__device__ U isce::cuda::core::gpuBicubicInterpolator<U>::interpolate(double x, double y, const U* z, size_t nx, size_t ny=0) {

    // The bicubic interpolation weights
    const double weights[] = {
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       -3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0,-2.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 0.0,-2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,-3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0,-2.0, 0.0, 0.0,-1.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,-2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
       -3.0, 3.0, 0.0, 0.0,-2.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,-3.0, 3.0, 0.0, 0.0,-2.0,-1.0, 0.0, 0.0,
        9.0,-9.0, 9.0,-9.0, 6.0, 3.0,-3.0,-6.0, 6.0,-6.0,-3.0, 3.0, 4.0, 2.0, 1.0, 2.0,
       -6.0, 6.0,-6.0, 6.0,-4.0,-2.0, 2.0, 4.0,-3.0, 3.0, 3.0,-3.0,-2.0,-1.0,-1.0,-2.0,
        2.0,-2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,-2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
       -6.0, 6.0,-6.0, 6.0,-3.0,-3.0, 3.0, 3.0,-4.0, 4.0, 2.0,-2.0,-2.0,-2.0,-1.0,-1.0,
        4.0,-4.0, 4.0,-4.0, 2.0, 2.0,-2.0,-2.0, 2.0,-2.0,-2.0, 2.0, 1.0, 1.0, 1.0, 1.0
    };

    size_t x1 = floor(x);
    size_t x2 = ceil(x);
    size_t y1 = floor(y);
    size_t y2 = ceil(y);

    const U denom = U(2.0);
    const U scale = U(0.25);

    // Future work: See "Future work" note from Interpolator::bilinear.
    const U zz[4] = {z[y1*nx + x1], z[y1*nx + x2], z[y2*nx + x2], z[y2*nx + x1]};

    // First order derivatives
    const U dzdx[4] = {
        (z[y1*nx + x1+1] - z[y1*nx + x1-1]) / denom,
        (z[y1*nx + x2+1] - z[y1*nx + x2-1]) / denom,
        (z[y2*nx + x2+1] - z[y2*nx + x2-1]) / denom,
        (z[y2*nx + x1+1] - z[y2*nx + x1-1]) / denom
    };
    const U dzdy[4] = {
        (z[(y1+1)*nx + x1] - z[(y1-1)*nx + x1]) / denom,
        (z[(y1+1)*nx + x2+1] - z[(y1-1)*nx + x2]) / denom,
        (z[(y2+1)*nx + x2+1] - z[(y2-1)*nx + x2]) / denom,
        (z[(y2+1)*nx + x1+1] - z[(y2-1)*nx + x1]) / denom
    };

    // Cross derivatives
    const U dzdxy[4] = {
        scale*(z[(y1+1)*nx + x1+1] - z[(y1-1)*nx + x1+1] - z[(y1+1)*nx + x1-1] + z[(y1-1)*nx + x1-1]),
        scale*(z[(y1+1)*nx + x2+1] - z[(y1-1)*nx + x2+1] - z[(y1+1)*nx + x2-1] + z[(y1-1)*nx + x2-1]),
        scale*(z[(y2+1)*nx + x2+1] - z[(y2-1)*nx + x2+1] - z[(y2+1)*nx + x2-1] + z[(y2-1)*nx + x2-1]),
        scale*(z[(y2+1)*nx + x1+1] - z[(y2-1)*nx + x1+1] - z[(y2+1)*nx + x1-1] + z[(y2-1)*nx + x1-1])
    };
      
    // Compute polynomial coefficients 
    U q[16];
    for (int i = 0; i < 4; ++i) {
        q[i] = zz[i];
        q[i+4] = dzdx[i];
        q[i+8] = dzdy[i];
        q[i+12] = dzdxy[i];
    }

    // Matrix multiply by stored weights
    U c[16];
    for (int i = 0; i < 16; ++i) {
        U qq(0.0);
        for (int j = 0; j < 16; ++j) {
            const U cpx_wt = (U)(weights[i*16+j]);
            qq += cpx_wt * q[j];
        }
        c[i] = qq;
    }

    // Compute and normalize desired results
    const U t = x - x1;
    const U u = y - y1;
    U ret = 0.0;
    for (int i = 3; i >= 0; i--) {
        ret = t*ret + ((c[i*4 + 3]*u + c[i*4 + 2])*u + c[i*4 + 1])*u + c[i*4];
    }
    return ret;
}


/*
 each template parameter needs it's own declaration here
 */
template class gpuBicubicInterpolator<double>;
template class gpuBicubicInterpolator<float>;

template __global__ void
gpuInterpolator_g<double>(gpuBicubicInterpolator<double> interp, double *x, double *y,
                          const double *z, double *value, size_t nx, size_t ny);

