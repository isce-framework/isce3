#pragma once

#include <isce/container/RadarGeometry.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce::container::RadarGeometry>&);
