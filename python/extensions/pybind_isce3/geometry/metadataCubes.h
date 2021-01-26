#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <isce3/geometry/metadataCubes.h>

void addbinding_metadata_cubes(pybind11::module&);
