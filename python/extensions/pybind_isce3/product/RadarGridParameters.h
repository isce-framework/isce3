#pragma once

#include <isce/product/RadarGridParameters.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce::product::RadarGridParameters> &);
