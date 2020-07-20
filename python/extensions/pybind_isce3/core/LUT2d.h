#pragma once

#include <isce3/core/LUT2d.h>
#include <pybind11/pybind11.h>

template<typename T>
void addbinding(pybind11::class_<isce3::core::LUT2d<T>> &);
