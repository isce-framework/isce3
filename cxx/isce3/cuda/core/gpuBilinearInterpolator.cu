#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <thrust/complex.h>

#include <isce3/core/Matrix.h>
#include <isce3/cuda/except/Error.h>

#include "gpuInterpolator.h"

using isce3::cuda::core::gpuBilinearInterpolator;
using isce3::cuda::core::gpuInterpolator;

template<class T>
__global__ void gpuInterpolator_g(gpuBilinearInterpolator<T> interp, double* x,
        double* y, const T* z, T* value, size_t nx, size_t ny = 0)
{
    /*
     *  GPU kernel to test interpolate() on the device for consistency.
     */
    int i = threadIdx.x;
    value[i] = interp.interpolate(x[i], y[i], z, nx, ny);
}

template<class T>
__host__ void isce3::cuda::core::gpuBilinearInterpolator<T>::interpolate_h(
        const Matrix<double>& truth, Matrix<T>& m, double start, double delta,
        T* h_z)
{
    /*
     *  CPU-side function to call the corresponding GPU function on a single
     * thread for consistency checking
     */

    // allocate host side memory
    size_t size_input_pts = truth.length() * sizeof(double);
    size_t size_output_pts = truth.length() * sizeof(T);
    size_t nx = m.width();

    double* h_x = (double*) malloc(size_input_pts);
    double* h_y = (double*) malloc(size_input_pts);

    // assign host side inputs
    for (size_t i = 0; i < truth.length(); ++i) {
        h_x[i] = (truth(i, 0) - start) / delta;
        h_y[i] = (truth(i, 1) - start) / delta;
    }

    // allocate devie side memory
    double* d_x;
    checkCudaErrors(cudaMalloc((void**) &d_x, size_input_pts));
    double* d_y;
    checkCudaErrors(cudaMalloc((void**) &d_y, size_input_pts));
    T* d_z;
    checkCudaErrors(cudaMalloc((void**) &d_z, size_output_pts));
    T* d_m;
    checkCudaErrors(cudaMalloc((T**) &d_m, m.length() * m.width() * sizeof(T)));

    // copy input data
    checkCudaErrors(
            cudaMemcpy(d_x, h_x, size_input_pts, cudaMemcpyHostToDevice));
    checkCudaErrors(
            cudaMemcpy(d_y, h_y, size_input_pts, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_m, m.data(),
            m.length() * m.width() * sizeof(T), cudaMemcpyHostToDevice));

    // launch!
    int n_threads = truth.length();
    gpuInterpolator_g<T><<<1, n_threads>>>(*this, d_x, d_y, d_m, d_z, nx);

    // copy device results to host
    checkCudaErrors(
            cudaMemcpy(h_z, d_z, size_output_pts, cudaMemcpyDeviceToHost));

    // free memory
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(d_m));
}

template<class T>
__device__ T isce3::cuda::core::gpuBilinearInterpolator<T>::interpolate(
        double x, double y, const T* z, size_t nx, size_t ny = 0)
{
    size_t x1 = floor(x);
    size_t x2 = ceil(x);
    size_t y1 = floor(y);
    size_t y2 = ceil(y);

    T q11 = z[y1 * nx + x1];
    T q12 = z[y2 * nx + x1];
    T q21 = z[y1 * nx + x2];
    T q22 = z[y2 * nx + x2];

    if ((y1 == y2) && (x1 == x2)) {
        return q11;
    } else if (y1 == y2) {
        return ((T)((x2 - x) / (x2 - x1)) * q11) +
               ((T)((x - x1) / (x2 - x1)) * q21);
    } else if (x1 == x2) {
        return ((T)((y2 - y) / (y2 - y1)) * q11) +
               ((T)((y - y1) / (y2 - y1)) * q12);
    } else {
        return ((q11 * (T)((x2 - x) * (y2 - y))) / (T)((x2 - x1) * (y2 - y1))) +
               ((q21 * (T)((x - x1) * (y2 - y))) / (T)((x2 - x1) * (y2 - y1))) +
               ((q12 * (T)((x2 - x) * (y - y1))) / (T)((x2 - x1) * (y2 - y1))) +
               ((q22 * (T)((x - x1) * (y - y1))) / (T)((x2 - x1) * (y2 - y1)));
    }
}

// Explicit instantiations
template class gpuBilinearInterpolator<double>;
template class gpuBilinearInterpolator<thrust::complex<double>>;
template class gpuBilinearInterpolator<float>;
template class gpuBilinearInterpolator<thrust::complex<float>>;
template class gpuBilinearInterpolator<unsigned char>;
template class gpuBilinearInterpolator<unsigned short>;
template class gpuBilinearInterpolator<unsigned int>;

template __global__ void gpuInterpolator_g<double>(
        gpuBilinearInterpolator<double> interp, double* x, double* y,
        const double* z, double* value, size_t nx, size_t ny);
