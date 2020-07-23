#pragma once

#include <isce3/geometry/Geo2rdr.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce3::geometry::Geo2rdr>&);
