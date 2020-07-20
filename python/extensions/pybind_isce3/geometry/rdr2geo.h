#pragma once

#include <isce/geometry/Topo.h>
#include <pybind11/pybind11.h>

void addbinding_rdr2geo(pybind11::module& m);
void addbinding(pybind11::class_<isce::geometry::Topo>&);
