// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2019
//

#include "gpuLUT1d.h"
#include "../helper_cuda.h"


/** Kernel for initializing interpolation object. */
__global__
void initInterpKernel(isce::cuda::core::gpuInterpolator<float> ** interp,
                      isce::core::dataInterpMethod interpMethod) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (interpMethod == isce::core::BILINEAR_METHOD) {
            (*interp) = new isce::cuda::core::gpuBilinearInterpolator<float>();
        } else if (interpMethod == isce::core::BICUBIC_METHOD) {
            (*interp) = new isce::cuda::core::gpuBicubicInterpolator<float>();
        } else if (interpMethod == isce::core::BIQUINTIC_METHOD) {
            (*interp) = new isce::cuda::core::gpuSpline2dInterpolator<float>(6); 
        } else {
            (*interp) = new isce::cuda::core::gpuBilinearInterpolator<float>();
        }
    }
}

/** Initialize interpolation object on device. */
template <typename T>
__host__
void isce::cuda::core::gpuLUT2d<T>::
initInterp() {
    // Allocate interpolator pointer on device
    checkCudaErrors(cudaMalloc(&_interp, sizeof(isce::cuda::core::gpuInterpolator<float> **)));

    // Call initialization kernel
    initInterpKernel<<<1, 1>>>(_interp, _interpMethod);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());
}


/** Kernel for deleting interpolation objects on device. */
__global__
void finalizeInterpKernel(isce::cuda::core::gpuInterpolator<float> ** interp) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *interp;
    }
}

/** Finalize/delete interpolation object on device. */
template <typename T>
__host__
void isce::cuda::core::gpuLUT2d<T>::
finalizeInterp() {
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
    _interpMethod(lut.interpMethod()) {

    // If input LUT2d does not have data, do not send anything to the device
    if (!lut.haveData()) {
        return;
    }

    // Allocate memory on device for LUT data
    size_t N = lut.length() * lut.width();
    checkCudaErrors(cudaMalloc((T **) &_data, N * sizeof(T)));

    // Copy LUT data
    checkCudaErrors(cudaMemcpy(_data, &lut.data().data(), N * sizeof(T),
                               cudaMemcpyHostToDevice));

    // Create interpolator
    initInterp();
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
        finalizeInterp();
    }
}
