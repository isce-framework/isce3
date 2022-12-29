#pragma once

#include <isce3/geogrid/getRadarGrid.h>
#include <pybind11/pybind11.h>

void addbinding_get_radar_grid(pybind11::module& m);
