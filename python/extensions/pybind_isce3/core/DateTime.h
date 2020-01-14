#pragma once

#include <isce/core/DateTime.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce::core::DateTime> &);
