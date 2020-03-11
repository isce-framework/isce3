#pragma once

#include <isce/core/Ellipsoid.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce::core::Ellipsoid> &);
