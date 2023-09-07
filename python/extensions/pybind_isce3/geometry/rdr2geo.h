#pragma once

#include <isce3/geometry/detail/Rdr2Geo.h>
#include <isce3/geometry/Topo.h>
#include <pybind11/pybind11.h>

void addbinding(pybind11::class_<isce3::geometry::detail::Rdr2GeoParams>&);
void addbinding_rdr2geo(pybind11::module& m);
void addbinding(pybind11::class_<isce3::geometry::Topo>&);
