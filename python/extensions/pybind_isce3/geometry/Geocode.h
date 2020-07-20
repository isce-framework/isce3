#pragma once

#include <isce3/geometry/Geocode.h>
#include <pybind11/pybind11.h>

template<typename T>
void addbinding(pybind11::class_<isce3::geometry::Geocode<T>>&);
void addbinding(pybind11::enum_<isce3::geometry::geocodeMemoryMode> &);
void addbinding(pybind11::enum_<isce3::geometry::geocodeOutputMode> &);
