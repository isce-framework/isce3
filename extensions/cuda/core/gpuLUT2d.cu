// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2019
//

#include <isce/core/Matrix.h>
#include "gpuLUT2d.h"
#include "../helper_cuda.h"

__device__
double clamp(double d, double min, double max) {
  const double t = d < min ? min : d;
  return t > max ? max : t;
}

/** Kernel for initializing interpolation object. */
template <typename T>
__global__
void initInterpKernel(isce::cuda::core::gpuInterpolator<T> ** interp,
                      isce::core::dataInterpMethod interpMethod) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (interpMethod == isce::core::BILINEAR_METHOD) {
            (*interp) = new isce::cuda::core::gpuBilinearInterpolator<T>();
        } else if (interpMethod == isce::core::BICUBIC_METHOD) {
            (*interp) = new isce::cuda::core::gpuBicubicInterpolator<T>();
        } else if (interpMethod == isce::core::BIQUINTIC_METHOD) {
            (*interp) = new isce::cuda::core::gpuSpline2dInterpolator<T>(6); 
        } else {
            (*interp) = new isce::cuda::core::gpuBilinearInterpolator<T>();
        }
    }
}

/** Initialize interpolation object on device. */
template <typename T>
__host__
void isce::cuda::core::gpuLUT2d<T>::
_initInterp() {
    // Allocate interpolator pointer on device
    checkCudaErrors(cudaMalloc(&_interp, sizeof(isce::cuda::core::gpuInterpolator<T> **)));

    // Call initialization kernel
    initInterpKernel<<<1, 1>>>(_interp, _interpMethod);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());
}


/** Kernel for deleting interpolation objects on device. */
template <typename T>
__global__
void finalizeInterpKernel(isce::cuda::core::gpuInterpolator<T> ** interp) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *interp;
    }
}

/** Finalize/delete interpolation object on device. */
template <typename T>
__host__
void isce::cuda::core::gpuLUT2d<T>::
_finalizeInterp() {
    // Call finalization kernel
    finalizeInterpKernel<<<1, 1>>>(_interp);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());

    // Free memory for pointers
    checkCudaErrors(cudaFree(_interp));
}

// Deep copy constructor from CPU LUT2d
/** @param[in] lut LUT2d<T> object */
template <typename T>
__host__
isce::cuda::core::gpuLUT2d<T>::
gpuLUT2d(const isce::core::LUT2d<T> & lut) :
    _haveData(lut.haveData()),
    _boundsError(lut.boundsError()),
    _refValue(lut.refValue()),
    _xstart(lut.xStart()),
    _ystart(lut.yStart()),
    _dx(lut.xSpacing()),
    _dy(lut.ySpacing()),
    _length(lut.length()),
    _width(lut.width()),
    _interpMethod(lut.interpMethod()) {

    // If input LUT2d does not have data, do not send anything to the device
    if (!lut.haveData()) {
        return;
    }

    // Allocate memory on device for LUT data
    size_t N = lut.length() * lut.width();
    checkCudaErrors(cudaMalloc((T **) &_data, N * sizeof(T)));

    // Copy LUT data
    const isce::core::Matrix<T> & lutData = lut.data();
    checkCudaErrors(cudaMemcpy(_data, lutData.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    // Create interpolator
    _initInterp();
    _owner = true;
}

// Shallow copy constructor on device
/** @param[in] lut gpuLUT2d<T> object */
template <typename T>
__host__ __device__
isce::cuda::core::gpuLUT2d<T>::
gpuLUT2d(isce::cuda::core::gpuLUT2d<T> & lut) :
    _haveData(lut.haveData()),
    _boundsError(lut.boundsError()),
    _refValue(lut.refValue()),
    _xstart(lut.xStart()),
    _ystart(lut.yStart()),
    _dx(lut.xSpacing()),
    _dy(lut.ySpacing()),
    _length(lut.length()),
    _width(lut.width()),
    _data(lut.data()),
    _interp(lut.interp()),
    _owner(false) {}

// Shallow assignment operator on device
/** @param[in] lut gpuLUT2d<T> object */
template <typename T>
__host__ __device__
isce::cuda::core::gpuLUT2d<T> &
isce::cuda::core::gpuLUT2d<T>::
operator=(isce::cuda::core::gpuLUT2d<T> & lut) {
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
template <typename T>
isce::cuda::core::gpuLUT2d<T>::
~gpuLUT2d() {
    // Only owner of memory clears it
    if (_owner && _haveData) {
        checkCudaErrors(cudaFree(_data));
        _finalizeInterp();
    }
}

// Evaluate LUT at coordinate
/** @param[in] y Y-coordinate for evaluation
  * @param[in] x X-coordinate for evaluation
  * @param[out] value Interpolated value */
template <typename T>
__device__
T isce::cuda::core::gpuLUT2d<T>::
eval(double y, double x) const {
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

template <typename T>
__global__
void eval_d(isce::cuda::core::gpuLUT2d<T> lut, double az, double rng, T * val) {
    *val = lut.eval(az, rng);
}

template <typename T>
__host__
T
isce::cuda::core::gpuLUT2d<T>::
eval_h(double az, double rng) {

    T * val_d;
    T val_h = 0.0;

    // Allocate memory for result on device
    checkCudaErrors(cudaMalloc((T **) &val_d, sizeof(T)));

    // Call the kernel with a single thread
    dim3 grid(1), block(1);
    eval_d<<<grid, block>>>(*this, az, rng, val_d);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());

    // Copy results from device to host
    checkCudaErrors(cudaMemcpy(&val_h, val_d, sizeof(T), cudaMemcpyDeviceToHost));

    // Clean up
    checkCudaErrors(cudaFree(val_d));
    return val_h;
}

// Forward declaration
template class isce::cuda::core::gpuLUT2d<double>;
template class isce::cuda::core::gpuLUT2d<float>;

// end of file  
