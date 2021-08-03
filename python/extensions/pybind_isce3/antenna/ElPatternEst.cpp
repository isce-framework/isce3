#include "ElPatternEst.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include <Eigen/Dense>

#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Linspace.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly1d.h>
#include <isce3/geometry/DEMInterpolator.h>

// Alias
namespace py = pybind11;
using namespace isce3::core;
using isce3::antenna::ElPatternEst;
using isce3::geometry::DEMInterpolator;

void addbinding(py::class_<ElPatternEst>& pyElPatternEst)
{
    pyElPatternEst
            // constructors
            .def(py::init([](double sr_start, const Orbit& orbit,
                                  int polyfit_deg = 6,
                                  const DEMInterpolator& dem_interp = {},
                                  double win_ped = 0.0,
                                  const Ellipsoid& ellips = {},
                                  bool center_scale_pf = false) {
                return ElPatternEst(sr_start, orbit, polyfit_deg, dem_interp,
                        win_ped, ellips, center_scale_pf);
            }),
                    py::arg("sr_start"), py::arg("orbit"),
                    py::arg("polyfit_deg") = 6,
                    py::arg_v("dem_interp", DEMInterpolator(), "0.0"),
                    py::arg("win_ped") = 0.0,
                    py::arg_v("ellips", Ellipsoid(), "WGS84"),
                    py::arg("center_scale_pf") = false)
            .def(py::init([](double sr_start, const Orbit& orbit,
                                  const DEMInterpolator& dem_interp) {
                return ElPatternEst(sr_start, orbit, 6, dem_interp, 0.0,
                        Ellipsoid(), false);
            }),
                    py::arg("sr_start"), py::arg("orbit"),
                    py::arg("dem_interp"))

            // properties
            .def_property_readonly("win_ped", &ElPatternEst::winPed)
            .def_property_readonly("polyfit_deg", &ElPatternEst::polyfitDeg)
            .def_property_readonly("is_center_scale_polyfit",
                    &ElPatternEst::isCenterScalePolyfit)

            // methods
            .def("power_pattern_2way", &ElPatternEst::powerPattern2way,
                    py::arg("echo_mat"), py::arg("sr_spacing"),
                    py::arg("chirp_rate"), py::arg("chirp_dur"),
                    py::arg("az_time") = std::nullopt, py::arg("size_avg") = 8,
                    py::arg("inc_corr") = true)
            .def("power_pattern_1way", &ElPatternEst::powerPattern1way,
                    py::arg("echo_mat"), py::arg("sr_spacing"),
                    py::arg("chirp_rate"), py::arg("chirp_dur"),
                    py::arg("az_time") = std::nullopt, py::arg("size_avg") = 8,
                    py::arg("inc_corr") = true)
            .doc() = R"(
A class for estimating one-way or two-way elevation (EL) power 
pattern from 2-D raw echo data over quasi-homogenous scene 
and provide bunch of meta data. The final power in dB scale is 
fit into N-order polynomials as a function of look angle in radians.
)";
}
