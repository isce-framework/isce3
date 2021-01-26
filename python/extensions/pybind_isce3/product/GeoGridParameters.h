#pragma once

#include <isce3/product/GeoGridParameters.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce3::product::GeoGridParameters> &);
void addbinding_bbox_to_geogrid(pybind11::module & m);
