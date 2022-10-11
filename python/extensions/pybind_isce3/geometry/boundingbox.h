#pragma once

#include <isce3/geometry/boundingbox.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce3::geometry::RadarGridBoundingBox> &);
void addbinding_boundingbox(pybind11::module& m);
