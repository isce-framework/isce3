#pragma once

#include <isce3/geogrid/relocateRaster.h>
#include <pybind11/pybind11.h>

void addbinding_relocate_raster(pybind11::module& m);