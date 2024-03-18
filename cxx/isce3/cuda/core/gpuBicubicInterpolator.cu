#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <thrust/complex.h>

#include <isce3/core/Matrix.h>
#include <isce3/cuda/except/Error.h>

#include "gpuInterpolator.h"

using isce3::cuda::core::gpuBicubicInterpolator;
using isce3::cuda::core::gpuInterpolator;

template<class T>
__global__ void gpuInterpolator_g(gpuBicubicInterpolator<T> interp, double* x,
        double* y, const T* z, T* value, size_t nx, size_t ny = 0)
{
    /*
     *  GPU kernel to test interpolate() on the device for consistency.
     */
    int i = threadIdx.x;
    value[i] = interp.interpolate(x[i], y[i], z, nx, ny);
}

template<class T>
__host__ void isce3::cuda::core::gpuBicubicInterpolator<T>::interpolate_h(
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
__device__ T isce3::cuda::core::gpuBicubicInterpolator<T>::interpolate(
        double x, double y, const T* z, size_t nx, size_t ny = 0)
{

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

    const T denom = T(2.0);
    const T scale = T(0.25);

    // Future work: See "Future work" note from Interpolator::bilinear.
    const T zz[4] = {
            z[y1 * nx + x1], z[y1 * nx + x2], z[y2 * nx + x2], z[y2 * nx + x1]};

    // First order derivatives
    const T dzdx[4] = {(z[y1 * nx + x1 + 1] - z[y1 * nx + x1 - 1]) / denom,
            (z[y1 * nx + x2 + 1] - z[y1 * nx + x2 - 1]) / denom,
            (z[y2 * nx + x2 + 1] - z[y2 * nx + x2 - 1]) / denom,
            (z[y2 * nx + x1 + 1] - z[y2 * nx + x1 - 1]) / denom};
    const T dzdy[4] = {(z[(y1 + 1) * nx + x1] - z[(y1 - 1) * nx + x1]) / denom,
            (z[(y1 + 1) * nx + x2 + 1] - z[(y1 - 1) * nx + x2]) / denom,
            (z[(y2 + 1) * nx + x2 + 1] - z[(y2 - 1) * nx + x2]) / denom,
            (z[(y2 + 1) * nx + x1 + 1] - z[(y2 - 1) * nx + x1]) / denom};

    // Cross derivatives
    const T dzdxy[4] = {
            scale * (z[(y1 + 1) * nx + x1 + 1] - z[(y1 - 1) * nx + x1 + 1] -
                            z[(y1 + 1) * nx + x1 - 1] +
                            z[(y1 - 1) * nx + x1 - 1]),
            scale * (z[(y1 + 1) * nx + x2 + 1] - z[(y1 - 1) * nx + x2 + 1] -
                            z[(y1 + 1) * nx + x2 - 1] +
                            z[(y1 - 1) * nx + x2 - 1]),
            scale * (z[(y2 + 1) * nx + x2 + 1] - z[(y2 - 1) * nx + x2 + 1] -
                            z[(y2 + 1) * nx + x2 - 1] +
                            z[(y2 - 1) * nx + x2 - 1]),
            scale * (z[(y2 + 1) * nx + x1 + 1] - z[(y2 - 1) * nx + x1 + 1] -
                            z[(y2 + 1) * nx + x1 - 1] +
                            z[(y2 - 1) * nx + x1 - 1])};

    // Compute polynomial coefficients
    T q[16];
    for (int i = 0; i < 4; ++i) {
        q[i] = zz[i];
        q[i + 4] = dzdx[i];
        q[i + 8] = dzdy[i];
        q[i + 12] = dzdxy[i];
    }

    // Matrix multiply by stored weights
    T c[16];
    for (int i = 0; i < 16; ++i) {
        T qq(0.0);
        for (int j = 0; j < 16; ++j) {
            const T cpx_wt = (T)(weights[i * 16 + j]);
            qq += cpx_wt * q[j];
        }
        c[i] = qq;
    }

    // Compute and normalize desired results
    const T t = x - x1;
    const T u = y - y1;
    T ret = 0.0;
    for (int i = 3; i >= 0; i--) {
        ret = t * ret +
              ((c[i * 4 + 3] * u + c[i * 4 + 2]) * u + c[i * 4 + 1]) * u +
              c[i * 4];
    }
    return ret;
}

// Explicit instantiations
template class gpuBicubicInterpolator<double>;
template class gpuBicubicInterpolator<thrust::complex<double>>;
template class gpuBicubicInterpolator<float>;
template class gpuBicubicInterpolator<thrust::complex<float>>;
template class gpuBicubicInterpolator<unsigned char>;
template class gpuBicubicInterpolator<unsigned short>;
template class gpuBicubicInterpolator<unsigned int>;

template __global__ void gpuInterpolator_g<double>(
        gpuBicubicInterpolator<double> interp, double* x, double* y,
        const double* z, double* value, size_t nx, size_t ny);
