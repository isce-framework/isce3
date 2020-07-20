#pragma once

#include <isce3/core/Quaternion.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce::core::Quaternion> &);
