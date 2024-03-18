#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <thrust/complex.h>

#include <isce3/core/Matrix.h>
#include <isce3/cuda/except/Error.h>

#include "gpuInterpolator.h"

#define MAX_ORDER 20

using isce3::cuda::core::gpuInterpolator;
using isce3::cuda::core::gpuSpline2dInterpolator;

template<class T>
__global__ void gpuInterpolator_g(gpuSpline2dInterpolator<T> interp, double* x,
        double* y, const T* z, T* value, size_t nx, size_t ny = 0)
{
    /*
     *  GPU kernel to test interpolate() on the device for consistency.
     */
    int i = threadIdx.x;
    value[i] = interp.interpolate(x[i], y[i], z, nx, ny);
}

template<class T>
__host__ void gpuSpline2dInterpolator<T>::interpolate_h(
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
    double* h_x = (double*) malloc(size_input_pts);
    double* h_y = (double*) malloc(size_input_pts);
    size_t nx = m.width();
    size_t ny = m.length();

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
    gpuInterpolator_g<T><<<1, n_threads>>>(*this, d_x, d_y, d_m, d_z, nx, ny);

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
__device__ void _initSpline(T* Y, int n, T* R, T* Q)
{
    Q[0] = 0.0;
    R[0] = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        const auto p = 1.0 / (0.5 * Q[i - 1] + 2.0);
        Q[i] = -0.5 * p;
        R[i] = (3 * (Y[i + 1] - 2 * Y[i] + Y[i - 1]) - 0.5 * R[i - 1]) * p;
    }
    R[n - 1] = 0.0;
    for (int i = (n - 2); i > 0; --i)
        R[i] = Q[i] * R[i + 1] + R[i];
}

template<class T>
__device__ T _spline(double x, T* Y, int n, T* R)
{
    if (x < 1.0) {
        return Y[0] + (x - 1.0) * (Y[1] - Y[0] - (R[1] / 6.0));
    } else if (x > n) {
        return Y[n - 1] + ((x - n) * (Y[n - 1] - Y[n - 2] + (R[n - 2] / 6.)));
    } else {
        int j = int(floor(x));
        auto xx = x - j;
        auto t0 = Y[j] - Y[j - 1] - (R[j - 1] / 3.0) - (R[j] / 6.0);
        auto t1 = xx * ((R[j - 1] / 2.0) + (xx * ((R[j] - R[j - 1]) / 6)));
        return Y[j - 1] + (xx * (t0 + t1));
    }
}

template<class T>
__device__ T gpuSpline2dInterpolator<T>::interpolate(
        double x, double y, const T* z, size_t nx, size_t ny = 0)
{
    // Get coordinates of start of spline window
    int i0, j0;

    if ((_order % 2) != 0) {
        i0 = y - 0.5;
        j0 = x - 0.5;
    } else {
        i0 = y;
        j0 = x;
    }
    i0 = i0 - (_order / 2) + 1;
    j0 = j0 - (_order / 2) + 1;

    T A[MAX_ORDER] = {0}, R[MAX_ORDER] = {0}, Q[MAX_ORDER] = {0},
      HC[MAX_ORDER] = {0};

    for (int i = 0; i < _order; ++i) {
        const int indi = min(max(i0 + i, 0), int(ny) - 2);
        for (int j = 0; j < _order; ++j) {
            const int indj = min(max(j0 + j, 0), int(nx) - 2);
            A[j] = z[(indi + 1) * nx + indj + 1];
        }
        _initSpline(A, _order, R, Q);
        HC[i] = _spline(x - j0, A, _order, R);
    }

    _initSpline(HC, _order, R, Q);
    T spline_out = (T)(_spline(y - i0, HC, _order, R));

    return spline_out;
}

// Explicit instantiation
template class gpuSpline2dInterpolator<double>;
template class gpuSpline2dInterpolator<thrust::complex<double>>;
template class gpuSpline2dInterpolator<float>;
template class gpuSpline2dInterpolator<thrust::complex<float>>;
template class gpuSpline2dInterpolator<unsigned char>;
template class gpuSpline2dInterpolator<unsigned short>;
template class gpuSpline2dInterpolator<unsigned int>;

template __global__ void gpuInterpolator_g<double>(
        gpuSpline2dInterpolator<double> interp, double* x, double* y,
        const double* z, double* value, size_t nx, size_t ny);
