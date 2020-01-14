#pragma once

#include <isce/core/TimeDelta.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce::core::TimeDelta> &);
