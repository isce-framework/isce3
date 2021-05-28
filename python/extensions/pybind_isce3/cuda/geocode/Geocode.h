#pragma once

#include <isce3/cuda/geocode/Geocode.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce3::cuda::geocode::Geocode>&);
