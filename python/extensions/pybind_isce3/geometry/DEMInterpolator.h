#pragma once

#include <isce/geometry/DEMInterpolator.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce::geometry::DEMInterpolator>&);
