#pragma once

#include <isce3/geocode/GeocodePolygon.h>
#include <pybind11/pybind11.h>

template<typename T>
void addbinding(pybind11::class_<isce3::geocode::GeocodePolygon<T>>&);