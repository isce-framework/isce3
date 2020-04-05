#pragma once

#include <isce/core/Interp1d.h>
#include <pybind11/pybind11.h>
#include <string>

template <typename T>
void addbinding_interp1d(pybind11::module & m, const char * name);
