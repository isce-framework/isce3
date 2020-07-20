#include "gpuLUT2d.h"

#include <isce3/core/LUT2d.h>
#include <isce3/core/Matrix.h>
#include <isce3/cuda/core/gpuInterpolator.h>
#include <isce3/cuda/except/Error.h>

namespace isce { namespace cuda { namespace core {

__device__ double clamp(double d, double min, double max)
{
    const double t = d < min ? min : d;
    return t > max ? max : t;
}

// Kernel for initializing interpolation object.
template<typename T>
__global__ void initInterpKernel(gpuInterpolator<T>** interp,
                                 isce::core::dataInterpMethod interpMethod)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (interpMethod == isce::core::BILINEAR_METHOD) {
            (*interp) = new gpuBilinearInterpolator<T>();
        } else if (interpMethod == isce::core::BICUBIC_METHOD) {
            (*interp) = new gpuBicubicInterpolator<T>();
        } else if (interpMethod == isce::core::BIQUINTIC_METHOD) {
            (*interp) = new gpuSpline2dInterpolator<T>(6);
        } else {
            (*interp) = new gpuBilinearInterpolator<T>();
        }
    }
}

// Initialize interpolation object on device.
template<typename T>
void gpuLUT2d<T>::_initInterp()
{
    // Allocate interpolator pointer on device
    checkCudaErrors(cudaMalloc(&_interp, sizeof(gpuInterpolator<T>**)));

    // Call initialization kernel
    initInterpKernel<<<1, 1>>>(_interp, _interpMethod);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
}

// Kernel for deleting interpolation objects on device.
template<typename T>
__global__ void finalizeInterpKernel(gpuInterpolator<T>** interp)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *interp;
    }
}

// Finalize/delete interpolation object on device.
template<typename T>
void gpuLUT2d<T>::_finalizeInterp()
{
    // Call finalization kernel
    finalizeInterpKernel<<<1, 1>>>(_interp);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());

    // Free memory for pointers
    checkCudaErrors(cudaFree(_interp));
}

// Deep copy constructor from CPU LUT2d
template<typename T>
gpuLUT2d<T>::gpuLUT2d(const isce::core::LUT2d<T>& lut)
    : _haveData(lut.haveData()), _boundsError(lut.boundsError()),
      _refValue(lut.refValue()), _xstart(lut.xStart()), _ystart(lut.yStart()),
      _dx(lut.xSpacing()), _dy(lut.ySpacing()), _length(lut.length()),
      _width(lut.width()), _interpMethod(lut.interpMethod())
{

    // If input LUT2d does not have data, do not send anything to the device
    if (!lut.haveData()) {
        return;
    }

    // Allocate memory on device for LUT data
    size_t N = lut.length() * lut.width();
    checkCudaErrors(cudaMalloc((T**) &_data, N * sizeof(T)));

    // Copy LUT data
    const isce::core::Matrix<T>& lutData = lut.data();
    checkCudaErrors(cudaMemcpy(_data, lutData.data(), N * sizeof(T),
                               cudaMemcpyHostToDevice));

    // Create interpolator
    _initInterp();
    _owner = true;
}

template<typename T>
gpuLUT2d<T>::gpuLUT2d(const gpuLUT2d<T>& other)
    : _haveData(other.haveData()), _boundsError(other.boundsError()),
      _refValue(other.refValue()), _xstart(other.xStart()),
      _ystart(other.yStart()), _dx(other.xSpacing()), _dy(other.ySpacing()),
      _length(other.length()), _width(other.width()),
      _interpMethod(other.interpMethod()), _owner(true)
{
    if (haveData()) {
        // allocate device storage for LUT data & copy
        size_t bytes = length() * width() * sizeof(T);
        checkCudaErrors(cudaMalloc(&_data, bytes));
        checkCudaErrors(cudaMemcpy(_data, other.data(), bytes,
                                   cudaMemcpyDeviceToDevice));

        _initInterp();
    }
}

// Shallow copy constructor on device
template<typename T>
__host__ __device__ gpuLUT2d<T>::gpuLUT2d(gpuLUT2d<T>& lut)
    : _haveData(lut.haveData()), _boundsError(lut.boundsError()),
      _refValue(lut.refValue()), _xstart(lut.xStart()), _ystart(lut.yStart()),
      _dx(lut.xSpacing()), _dy(lut.ySpacing()), _length(lut.length()),
      _width(lut.width()), _data(lut.data()), _interp(lut.interp()),
      _owner(false)
{}

// Shallow assignment operator on device
template<typename T>
__host__ __device__ gpuLUT2d<T>& gpuLUT2d<T>::operator=(gpuLUT2d<T>& lut)
{
    _haveData = lut.haveData();
    _boundsError = lut.boundsError();
    _refValue = lut.refValue();
    _xstart = lut.xStart();
    _ystart = lut.yStart();
    _dx = lut.xSpacing();
    _dy = lut.ySpacing();
    _length = lut.length();
    _width = lut.width();
    _data = lut.data();
    _interp = lut.interp();
    _owner = false;
    return *this;
}

// Destructor
template<typename T>
gpuLUT2d<T>::~gpuLUT2d()
{
    // Only owner of memory clears it
    if (_owner && _haveData) {
        checkCudaErrors(cudaFree(_data));
        _finalizeInterp();
    }
}

// Evaluate LUT at coordinate
template<typename T>
__device__ T gpuLUT2d<T>::eval(double y, double x) const
{
    /*
     * Evaluate the LUT at the given coordinates.
     */

    // Check if data are available; if not, return ref value
    T value = _refValue;
    if (!_haveData) {
        return value;
    }

    // Get matrix indices corresponding to requested coordinates
    double x_idx = (x - _xstart) / _dx;
    double y_idx = (y - _ystart) / _dy;

    // Check bounds or clamp indices to valid values
    if (_boundsError) {
        if (x_idx < 0.0 || y_idx < 0.0 || x_idx >= _width || y_idx >= _length) {
            return value;
        }
    } else {
        x_idx = clamp(x_idx, 0.0, _width - 1.0);
        y_idx = clamp(y_idx, 0.0, _length - 1.0);
    }

    // Call interpolator
    value = (*_interp)->interpolate(x_idx, y_idx, _data, _width, _length);
    return value;
}

template<typename T>
__global__ void eval_d(gpuLUT2d<T> lut, double az, double rng, T* val)
{
    *val = lut.eval(az, rng);
}

template<typename T>
T gpuLUT2d<T>::eval_h(double az, double rng)
{

    T* val_d;
    T val_h = 0.0;

    // Allocate memory for result on device
    checkCudaErrors(cudaMalloc((T**) &val_d, sizeof(T)));

    // Call the kernel with a single thread
    dim3 grid(1), block(1);
    eval_d<<<grid, block>>>(*this, az, rng, val_d);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());

    // Copy results from device to host
    checkCudaErrors(
            cudaMemcpy(&val_h, val_d, sizeof(T), cudaMemcpyDeviceToHost));

    // Clean up
    checkCudaErrors(cudaFree(val_d));
    return val_h;
}

// Forward declaration
template class gpuLUT2d<double>;
template class gpuLUT2d<float>;

}}} // namespace isce::cuda::core
