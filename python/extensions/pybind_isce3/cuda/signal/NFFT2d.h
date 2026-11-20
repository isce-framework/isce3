#pragma once

#include <pybind11/pybind11.h>
#include <isce3/cuda/signal/NFFT2d.h>

template<typename T>
void addbinding(pybind11::class_<isce3::cuda::signal::NFFT2d<T>>& pyNFFT2d);

template<typename T>
void addbinding(pybind11::class_<isce3::cuda::signal::NFFT2dResult<T>>& pyNFFT2dResult);

void addbinding_make_image_nfft2d_gpu(pybind11::module& m);