#pragma once

#include <isce/core/LookSide.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::enum_<isce::core::LookSide> &);
