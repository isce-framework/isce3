#include "lookIncFromSr.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Orbit.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/geometry.h>

// Aliases
namespace py = pybind11;
using isce3::geometry::DEMInterpolator;
using namespace isce3::core;

// Functions binding
void addbinding_look_inc_from_sr(pybind11::module& m)
{

    m.def(
            "look_inc_ang_from_slant_range",
            [](double slant_range, const Orbit& orbit,
                    std::optional<double> az_time = {},
                    const DEMInterpolator& dem_interp = {},
                    const Ellipsoid& ellips = {}) {
                return lookIncAngFromSlantRange(
                        slant_range, orbit, az_time, dem_interp, ellips);
            },
            py::arg("slant_range"), py::arg("orbit"),
            py::arg("az_time") = std::nullopt,
            py::arg_v("dem_interp", DEMInterpolator(), "0.0"),
            py::arg_v("ellips", Ellipsoid(), "WGS84"),
            R"(
Estimate look angle (off-nadir angle) and local incidence angle at a desired slant range 
from orbit(spacecraft/antenna statevector) and at a certain relative azimuth time.

Parameters
----------
slant_range : float
    true slant range in meters from antenna phase center (or spacecraft position) to the ground. 
orbit : isce3.core.orbit
az_time : float, optional 
    relative azimuth time in seconds w.r.t reference epoch time of orbit object.
    If not specified, the mid time of orbit will be used as azimuth time.
dem_interp : isce3.geometry.DEMInterpolator, default=0.0
ellips : isce3.core.Ellipsoid, default=WGS84

Returns
-------
float
    Look angle or off-nadir angle in (rad)
float 
    Incidence angle in (rad)

Raises
------
RuntimeError
    for bad-value look angle or incidence angles

Notes
-----
See references [1]_ and [2]_ for the equations to calculate 
look angle and incidence angle, respectivelty.

References
----------
..[1] https://en.wikipedia.org/wiki/Law_of_cosines
..[2] https://en.wikipedia.org/wiki/Law_of_sines
)");

    m.def(
            "look_inc_ang_from_slant_range",
            [](const Eigen::Ref<const Eigen::ArrayXd>& slant_range,
                    const Orbit& orbit, std::optional<double> az_time = {},
                    const DEMInterpolator& dem_interp = {},
                    const Ellipsoid& ellips = {}) {
                return lookIncAngFromSlantRange(
                        slant_range, orbit, az_time, dem_interp, ellips);
            },
            py::arg("slant_range"), py::arg("orbit"),
            py::arg("az_time") = std::nullopt,
            py::arg_v("dem_interp", DEMInterpolator(), "0.0"),
            py::arg_v("ellips", Ellipsoid(), "WGS84"),
            R"(
Estimate look angles (off-nadir angle) and local incidence angles at desired slant ranges
from orbit(spacecraft/antenna statevector) and at a certain relative azimuth time.

Parameters
----------
slant_range : numpy.ndarray(float)
    Array of slant ranges in meters from antenna phase center (or spacecraft position) to the ground. 
orbit : isce3.core.orbit
az_time : float, optional 
    relative azimuth time in seconds w.r.t reference epoch time of orbit object.
    If not specified, the mid time of orbit will be used as azimuth time.
dem_interp : isce3.geometry.DEMInterpolator, default=0.0
ellips : isce3.core.Ellipsoid, default=WGS84

Returns
-------
numpy.ndarray(float)
    Look angles or off-nadir angles in (rad)
numpy.ndarray(float) 
    Incidence angles in (rad)

Raises
------
RuntimeError
    for bad-value look angle or incidence angles

Notes
-----
See references [1]_ and [2]_ for the equations to calculate 
look angle and incidence angle, respectivelty.

References
----------
..[1] https://en.wikipedia.org/wiki/Law_of_cosines
..[2] https://en.wikipedia.org/wiki/Law_of_sines
)");
}
