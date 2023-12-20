#pragma once

#include <pybind11/pybind11.h>
#include <isce3/geometry/detail/Rdr2Geo.h>
#include <isce3/geometry/detail/Geo2Rdr.h>

void addbinding_backproject(pybind11::module& m);

isce3::geometry::detail::Rdr2GeoBracketParams
parse_rdr2geo_params(const pybind11::dict& params);

isce3::geometry::detail::Geo2RdrBracketParams
parse_geo2rdr_params(const pybind11::dict& params);
