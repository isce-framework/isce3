#pragma once

#include <pybind11/pybind11.h>
#include <isce3/signal/NFFT2d.h>

isce3::signal::NFFT2dParams parse_nfft2d_params(const pybind11::dict& params);

template<typename T>
void addbinding(pybind11::class_<isce3::signal::NFFT2d<T>>& pyNFFT2d);

void addbinding_make_image_nfft2d(pybind11::module& m);