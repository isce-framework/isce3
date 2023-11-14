#include "ElNullRangeEst.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <isce3/antenna/Frame.h>
#include <isce3/core/Attitude.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/EMatrix.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Linspace.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly1d.h>
#include <isce3/geometry/DEMInterpolator.h>

// Alias
namespace py = pybind11;
using namespace isce3::core;
using namespace isce3::antenna;
using isce3::geometry::DEMInterpolator;

// readonly struct or aggregate  datatypes
void addbinding(py::class_<NullProduct>& pyNullProduct)
{
    pyNullProduct.def_readonly("slant_range", &NullProduct::slant_range)
            .def_readonly("el_angle", &NullProduct::el_angle)
            .def_readonly("doppler", &NullProduct::doppler)
            .def_readonly("magnitude", &NullProduct::magnitude);
    pyNullProduct.doc() = R"(
EL null product

Attributes
----------
slant_range : float
    Slant range of the null location in (m).
el_angle : float
    Elevation angle of the null location in (rad).
doppler : float
    Doppler at the null location in (Hz).
magnitude : float
    Relative magnitude of the null w.r.t left/right peaks in (linear).
)";
}

void addbinding(py::class_<NullConvergenceFlags>& pyNullConvergenceFlags)
{
    pyNullConvergenceFlags
            .def_readonly("newton_solver", &NullConvergenceFlags::newton_solver)
            .def_readonly("geometry_echo", &NullConvergenceFlags::geometry_echo)
            .def_readonly("geometry_antenna",
                    &NullConvergenceFlags::geometry_antenna);
    pyNullConvergenceFlags.doc() = R"(
A set of flags indicating convergence of iterative operations
used in EL null product formation

Attributes
----------
newton_solver : bool
    Indicates convergence of the 1-D Newton root solver.
geometry_echo : bool
    Indicates geometry-related convergence for echo null estimation.
geometry_antenna : bool
    Indicates geometry-related convergence for antenna null estimation.
)";
}

void addbinding(py::class_<NullPowPatterns>& pyNullPowPatterns)
{
    pyNullPowPatterns.def_readonly("ant", &NullPowPatterns::ant)
            .def_readonly("echo", &NullPowPatterns::echo)
            .def_readonly("el", &NullPowPatterns::el);
    pyNullPowPatterns.doc() = R"(
Elevation (EL) null power patterns for both antenna and echo as a function of EL
angles. The null is formed from two adjacent beams/channels in EL direction.

Attributes
----------
ant : array of float
    1-D array of peak-normalized null power pattern (linear) formed from antenna
    EL-cuts. The same size as `el`.
echo : array of float
    1-D array of peak-normalized null power pattern (linear) formed from echo
    channels in EL direction. The same size as `el`.
el : array of float
    Array of EL angles (radians) in antenna EL-AZ frame.
)";
}

// class
void addbinding(py::class_<ElNullRangeEst>& pyElNullRangeEst)
{
    pyElNullRangeEst
            // constructors
            .def(py::init([](double wavelength, double sr_spacing,
                                  double chirp_rate, double chirp_dur,
                                  const Orbit& orbit, const Attitude& attitude,
                                  const DEMInterpolator& dem_interp = {},
                                  const Frame& ant_frame = {},
                                  const Ellipsoid& ellips = {},
                                  double el_res = 8.726646259971648e-06,
                                  double abs_tol_dem = 1.0,
                                  int max_iter_dem = 20, int polyfit_deg = 6) {
                return ElNullRangeEst(wavelength, sr_spacing, chirp_rate,
                        chirp_dur, orbit, attitude, dem_interp, ant_frame,
                        ellips, el_res, abs_tol_dem, max_iter_dem, polyfit_deg);
            }),
                    py::arg("wavelength"), py::arg("sr_spacing"),
                    py::arg("chirp_rate"), py::arg("chirp_dur"),
                    py::arg("orbit"), py::arg("attitude"),
                    py::arg_v("dem_interp", DEMInterpolator(), "0.0"),
                    py::arg_v("ant_frame", Frame(), "EL_AND_AZ"),
                    py::arg_v("ellips", Ellipsoid(), "WGS84"),
                    py::arg("el_res") = 8.726646259971648e-06,
                    py::arg("abs_tol_dem") = 1.0, py::arg("max_iter_dem") = 20,
                    py::arg("polyfit_deg") = 6)

            // methods
            .def("genNullRangeDoppler", &ElNullRangeEst::genNullRangeDoppler,
                    py::arg("echo_left"), py::arg("echo_right"),
                    py::arg("el_cut_left"), py::arg("el_cut_right"),
                    py::arg("sr_start"), py::arg("el_ang_start"),
                    py::arg("el_ang_step"), py::arg("az_ang"),
                    py::arg("az_time") = std::nullopt, R"(
Generate null products from echo (measured) and antenna (nominal/expected).
The null product consists of azimuth time tag, null relative magnitude and its
location in EL and slant range, plut its Doppler value given 
azimuth (antenna geometry)/squint(Radar geometry) angle.

Parameters
----------
echo_left : np.ndarray(complex64) 
    complex 2-D array of raw echo samples (pulse by range) for
    the left RX channel corresponding to the left beam.
echo_right : np.ndarray(complex64)
    complex 2-D array of raw echo samples (pulse by range) for
    the right RX channel corresponding to the right beam. 
    Must have the same shape as of that of left one!
el_cut_left : np.ndarray(complex128) 
    complex array of uniformly-sampled relative or absolute 
    EL-cut antenna pattern on the left side.
el_cut_right : np.ndarray(complex128)  
    complex array of uniformly-sampled relative or absolute 
    EL-cut antenna pattern on the right side. 
    It must have the same size as left one!
sr_start : float
    start slant range (m) for both uniformly-sampled echoes in range.
el_ang_start : float 
    start elevation angle for left/right EL patterns in (rad)
el_ang_step : float 
    step elevation angle for left/right EL patterns in (rad) 
az_ang : float 
    azimuth angle (antenna geometry) or squint angle (Radar geometry) 
    in (rad). This angle determines the final Doppler centroid on 
    top of slant range value for both echo and antenna nulls.
az_time : float, (optional) 
    azimuth time of echoes in (sec) w.r.t reference epoch of orbit. 
    If not specified, the mid azimuth time of orbit will be used instead.

Returns
-------
isce3::core::DateTime
    azimuth time tag of the null product
isce3::antenna::NullProduct 
    echo null product
isce3::antenna:NullProduct
    antenna null product
isce3::antenna::NullConvergenceFlags
    all flags indicating convergence of iterative operations used 
    in forming both echo and antenna EL null products.
isce3::antenna::NullPowPatterns
    null power patterns (linear) for both antenna and echo as
    a function of EL angles (radians).

Raises
------
ValueError
    for bad input arguments
RuntimeError
    for failure in null formation

)")

            // properties
            .def_property_readonly("wave_length", &ElNullRangeEst::waveLength)
            .def_property_readonly(
                    "slant_range_spacing", &ElNullRangeEst::slantRangeSpacing)
            .def_property_readonly(
                    "grid_type_name", &ElNullRangeEst::gridTypeName)
            .def_property_readonly(
                    "chirp_sample_ref", &ElNullRangeEst::chirpSampleRef)
            .def_property_readonly("ref_epoch", &ElNullRangeEst::refEpoch)
            .def_property_readonly(
                    "dem_ref_height", &ElNullRangeEst::demRefHeight)
            .def_property_readonly(
                    "mid_time_orbit", &ElNullRangeEst::midTimeOrbit)
            .def_property_readonly(
                    "max_el_spacing", &ElNullRangeEst::maxElSpacing)
            .def_property_readonly("atol_dem", &ElNullRangeEst::atolDEM)
            .def_property_readonly("max_iter_dem", &ElNullRangeEst::maxIterDEM)
            .def_property_readonly("atol_null", &ElNullRangeEst::atolNull)
            .def_property_readonly(
                    "max_iter_null", &ElNullRangeEst::maxIterNull)
            .def_property_readonly("polyfit_deg", &ElNullRangeEst::polyfitDeg)

            .doc() = R"(
A class for forming Null power patterns in EL direction from both
a pair of adjacent El-cut antenna patterns as well as the respective
raw echoes of two adjacent RX channels.
The location of null in both antenna and echo domain will be estimated
and their respective values in EL angle, slant range, and Doppler will
be reported at a specific azimuth time in orbit.
See the following link and its references for algorithm, simulation and analyses,
https://github.jpl.nasa.gov/NISAR-POINTING/DOC/blob/master/Null_ELPointEst_REEsim_RevB.pdf
)";
}
