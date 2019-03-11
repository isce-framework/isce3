// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#include "gpuLUT1d.h"
#include "../helper_cuda.h"

// Deep copy constructor from CPU LUT1d
/** @param[in] lut LUT1d<T> object */
template <typename T>
__host__
isce::cuda::core::gpuLUT1d<T>::
gpuLUT1d(const isce::core::LUT1d<T> & lut) :
    _extrapolate(lut.extrapolate()), _size(lut.size()), _owner(true) {

    // Check if LUT1d has data
    if (!lut.haveData()) {
        _haveData = false;
        _refValue = lut.refValue();
        return;
    }

    // Allocate memory on device for LUT data
    size_t N = lut.size();
    checkCudaErrors(cudaMalloc((double **) &_coords, N * sizeof(double)));
    checkCudaErrors(cudaMalloc((T **) &_values, N * sizeof(T)));

    // Copy LUT data
    checkCudaErrors(cudaMemcpy(_coords, &lut.coords()[0], N*sizeof(double),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_values, &lut.values()[0], N*sizeof(double),
                               cudaMemcpyHostToDevice));

    // Set flags if everything has been successful
    _haveData = true;
    _refValue = lut.values()[0];
}

// Shallow copy constructor on device
/** @param[in] lut gpuLUT1d<T> object */
template <typename T>
__host__ __device__
isce::cuda::core::gpuLUT1d<T>::
gpuLUT1d(isce::cuda::core::gpuLUT1d<T> & lut) :
    _haveData(lut.haveData()), _refValue(lut.refValue()),
    _coords(lut.coords()), _values(lut.values()),
    _size(lut.size()), _extrapolate(lut.extrapolate()),
    _owner(false) {}

// Shallow assignment operator on device
/** @param[in] lut gpuLUT1d<T> object */
template <typename T>
__host__ __device__
isce::cuda::core::gpuLUT1d<T> &
isce::cuda::core::gpuLUT1d<T>::
operator=(isce::cuda::core::gpuLUT1d<T> & lut) {
    _haveData = lut.haveData();
    _refValue = lut.refValue();
    _coords = lut.coords();
    _values = lut.values();
    _size = lut.size();
    _extrapolate = lut.extrapolate();
    _owner = false;
    return *this;
}

/** Destructor */
template <typename T>
isce::cuda::core::gpuLUT1d<T>::
~gpuLUT1d() {
    // Only owner of memory clears it
    if (_owner && _haveData) {
        checkCudaErrors(cudaFree(_coords));
        checkCudaErrors(cudaFree(_values));
    }
}

// Evaluate the LUT
/** @param[in] x Point to evaluate the LUT
  * @param[out] result Interpolated value */
template <typename T>
__device__
T isce::cuda::core::gpuLUT1d<T>::
eval(double x) const {

    // Check if data are available; if not, return ref value
    T result = _refValue;
    if (!_haveData) {
        return result;
    }

    // Check bounds to see if we need to perform linear extrapolation
    const int n = _size;
    if (x < _coords[0]) {
        if (_extrapolate) {
            const double dx = _coords[0] - _coords[1];
            const double dy = _values[0] - _values[1];
            const double d = x - _coords[1];
            result = (dy / dx) * d + _values[1];
            return result; 
        } else {
            return result;
        }
    } else if (x > _coords[n-1]) {
        if (_extrapolate) {
            const double dx = _coords[n-1] - _coords[n-2];
            const double dy = _values[n-1] - _values[n-2];
            const double d = x - _coords[n-2];
            result = (dy / dx) * d + _values[n-2];
            return result;
        } else {
            return result;
        }
    }

    // Otherwise, proceed with interpolation
    // Binary search to find leftmost coordinate
    int low = 0;
    int high = _size;
    while (low < high) {
        const int midpoint = (low + high) / 2;
        if (_coords[midpoint] < x) {
            low = midpoint + 1;
        } else { 
            high = midpoint;
        }
    }

    // Check if right on top of a coordinate
    if (abs(_coords[high] - x) < 1.0e-12) {
        return _values[high];
    }

    // The indices of the x bounds
    const int j0 = high - 1;
    const int j1 = high;
    
    // Get coordinates at bounds
    double x1 = _coords[j0];
    double x2 = _coords[j1];
    
    // Interpolate
    result = (x2 - x) / (x2 - x1) * _values[j0] + (x - x1) / (x2 - x1) * _values[j1];
    return result;
}

template <typename T>
__global__
void eval_d(isce::cuda::core::gpuLUT1d<T> lut, double rng, T * val) {
    *val = lut.eval(rng);
}

template <typename T>
__host__
T
isce::cuda::core::gpuLUT1d<T>::
eval_h(double rng) {

    T * val_d;
    T val_h;
    
    // Allocate memory for result on device
    checkCudaErrors(cudaMalloc((T **) &val_d, sizeof(T)));

    // Call the kernel with a single thread
    dim3 grid(1), block(1);
    eval_d<<<grid, block>>>(*this, rng, val_d);

    // Copy results from device to host
    checkCudaErrors(cudaMemcpy(&val_h, val_d, sizeof(T), cudaMemcpyDeviceToHost));

    // Clean up
    checkCudaErrors(cudaFree(val_d));
    return val_h;
}

// Forward declaration
template class isce::cuda::core::gpuLUT1d<double>;
template class isce::cuda::core::gpuLUT1d<float>;

// end of file  
