#pragma once

#include <pybind11/pybind11.h>
#include <isce3/focus/Backproject.h>
#include <isce3/geometry/detail/Rdr2Geo.h>
#include <isce3/geometry/detail/Geo2Rdr.h>

// Copied declaration from pybind_isce3/signal/NFFT2d.h
// Implementation is in pybind_isce3/signal/NFFT2d.cpp
isce3::signal::NFFT2dParams parse_nfft2d_params(const pybind11::dict& params);

void addbinding(pybind11::class_<isce3::focus::PolarGrid>& pyPolarGrid);
void addbinding_backproject(pybind11::module& m);

isce3::geometry::detail::Rdr2GeoBracketParams
parse_rdr2geo_params(const pybind11::dict& params);

isce3::geometry::detail::Geo2RdrBracketParams
parse_geo2rdr_params(const pybind11::dict& params);
