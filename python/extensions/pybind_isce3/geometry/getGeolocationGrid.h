#pragma once

#include <isce3/geometry/getGeolocationGrid.h>
#include <pybind11/pybind11.h>

void addbinding_get_geolocation_grid(pybind11::module& m);
