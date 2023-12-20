#include "rdr2geo_roots.h"

#include <pybind11/eigen.h>
#include <stdexcept>

#include <isce3/core/LookSide.h>
#include <isce3/core/Vector.h>
#include <isce3/core/Orbit.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/rdr2geo_roots.h>
#include <pybind_isce3/core/LookSide.h>

namespace py = pybind11;

using namespace isce3::core;
using namespace isce3::geometry;

void addbinding_rdr2geo_roots(py::module& m)
{
    m.def(
            "rdr2geo_bracket",
            [](double aztime, double slantRange, const Orbit& orbit,
                    py::object pySide, double doppler, double wavelength,
                    const DEMInterpolator& dem, double tolHeight,
                    double lookMin, double lookMax) {
                Vec3 targetXYZ;
                const auto side = duck_look_side(pySide);
                int success = rdr2geo_bracket(aztime, slantRange, doppler,
                        orbit, dem, targetXYZ, wavelength, side, tolHeight,
                        lookMin, lookMax);
                if (!success) {
                    throw std::runtime_error("failed to converge");
                }
                return targetXYZ;
            },
            py::arg("aztime"), py::arg("slant_range"), py::arg("orbit"),
            py::arg("side"), py::arg("doppler"), py::arg("wavelength"),
            py::arg("dem") = DEMInterpolator(),
            py::arg("tol_height") = isce3::geometry::detail::DEFAULT_TOL_HEIGHT,
            py::arg("look_min") = 0.0, py::arg("look_max") = M_PI / 2,
            R"(
            Convert radar coordinates to geographic coordinates.

            Parameters
            ----------
            aztime : float
                Azimuth time in seconds since orbit.reference_epoch
            slant_range : float
                Slant range to target in meters
            orbit : isce3.core.Orbit
                Reference orbit that defines the radar coordinate system.
            side : Union[isce3.core.LookSide, str]
                Radar look direction (LookSide.Left or LookSide.Right) or
                "left" or "right".
            doppler : float
                Reference Doppler frequency in Hz that defines the radar
                coordinate system.  Note that this isn't necessarily the
                processed Doppler centroid.  For example, NISAR products are
                generated on a zero-Doppler grid (doppler=0).
            wavelength : float
                Wavelength in meters corresponding to the provided Doppler.
                If doppler=0 then wavelength has no effect on the solution
                (as long as it is finite).
            dem : isce3.geometry.DEMInterpolator, optional
                Digital elevation model provding height in meters above the
                WGS84 ellipsoid.  Defaults to an object that returns zero
                everywhere.
            tol_height : float, optional
                Allowable height error (in meters) in solution.  There is no
                error in the aztime or range directions.
            look_min : float
                Minimum pseudo-look angle in radians, default=0
            look_max : float
                Maximum pseudo-look angle in radians, default=pi/2

            Returns
            -------
            xyz : numpy.ndarray
                Target ECEF XYZ position in meters.

            Notes
            -----
            The solution must be in the provided bracket [look_min, look_max]
            or it will fail with an exception.  The defaults bracket [0, pi/2]
            is reasonable unless the radar is flying lower than nearby terrain
            (e.g., a drone flying in a valley).

            Usually the look angle is defined as the angle between the line of
            sight vector and the nadir vector.  Here the pseudo-look angle is
            defined in a similar way, except it's with respect to the projection
            of the nadir vector into a plane perpendicular to the velocity.  For
            simplicity, we use the geocentric nadir definition.
            )");
}
