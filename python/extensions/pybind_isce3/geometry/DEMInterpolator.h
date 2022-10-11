#pragma once

#include <isce3/geometry/DEMInterpolator.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce3::geometry::DEMInterpolator>&);
void addbinding_DEM_raster2interpolator(pybind11::module&);
