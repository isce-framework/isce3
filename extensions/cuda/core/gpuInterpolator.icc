//
// Author: Liang Yu
// Copyright 2018
//

#include <cuda_runtime.h>
#include "gpuInterpolator.h"

using isce::cuda::core::gpuInterpolator;
using isce::cuda::core::gpuBilinearInterpolator;


template <class U>
__global__ void gpuInterpolator_g(gpuInterpolator<U> interp, double x, double y, const U *z, size_t nx, U *value) {
    /*
     *  GPU kernel to test interpolate() on the device for consistency.
     */
    *value = interp.interpolate(x, y, z, nx); 
}


template <class U>
__host__ void isce::cuda::core::gpuInterpolator<U>::interpolate_h(const Matrix<double>& truth, const Matrix<U>& m, double start, double delta, U* z) {
    /*
     *  CPU-side function to call the corresponding GPU function on a single thread for consistency checking
     */
    double x, y;
    U *m_unified, *z_d;
    size_t nx = m.width();

    // allocate  memory
    cudaMalloc((U**)&m_unified, m.length()*m.width()*sizeof(U));

    // initialize memory
    cudaMemcpy(m_unified, &m.data()[0], m.length()*m.width()*sizeof(U), cudaMemcpyHostToDevice); 

    for (size_t i = 0; i < truth.length(); ++i) {
        x = (truth(i,0) - start) / delta;
        y = (truth(i,1) - start) / delta;
        gpuInterpolator_g<U><<<1, 1>>>(*this, x, y, m_unified, nx, z_d);
        z[i] = z_d;
    }
    
    // wait for GPU to finish before host access
    cudaDeviceSynchronize();

    cudaMemcpy(z, z_d, truth.length()*sizeof(U), cudaMemcpyDeviceToHost); 

    // free memory
    cudaFree(z_d);
    cudaFree(m_unified);
}


template <class U>
__device__ U isce::cuda::core::gpuBilinearInterpolator<U>::interpolate(double x, double y, const U* z, size_t nx) {
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

