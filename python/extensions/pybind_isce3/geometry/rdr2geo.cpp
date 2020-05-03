#include "rdr2geo.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>

#include <isce/core/LookSide.h>
#include <isce/core/Vector.h>
#include <isce/geometry/DEMInterpolator.h>
#include <isce/geometry/geometry.h>

using namespace isce::core;
using namespace isce::geometry;
namespace py = pybind11;

void addbinding_rdr2geo(py::module& m)
{
    m.def(
            "rdr2geo_cone", // name matches old Cython version
            [](const Vec3& radarXYZ, const Vec3& axis, double angle,
               double range, const DEMInterpolator& dem, py::object pySide,
               double threshold, int maxIter, int extraIter) {
                // duck type side
                LookSide side;
                if (py::isinstance<py::str>(pySide)) {
                    auto s = pySide.cast<std::string>();
                    side = parseLookSide(s);
                } else {
                    side = pySide.cast<LookSide>();
                }
                Vec3 targetXYZ;
                int converged =
                        rdr2geo(radarXYZ, axis, angle, range, dem, targetXYZ,
                                side, threshold, maxIter, extraIter);
                if (!converged)
                    throw std::runtime_error("rdr2geo failed to converge");
                return targetXYZ;
            },
            py::arg("radar_xyz"), py::arg("axis"), py::arg("angle"),
            py::arg("range"), py::arg("dem"), py::arg("side"),
            py::arg("threshold") = 0.05, py::arg("maxiter") = 50,
            py::arg("extraiter") = 50, R"(
    Solve for target position given radar position, range, and cone angle.
    The cone is described by a generating axis and the complement of the angle
    to that axis (e.g., angle=0 means a plane perpendicular to the axis).  The
    vertex of the cone is at the radar position, as is the center of the range
    sphere.

    Typically `axis` is the velocity vector and `angle` is the squint angle.
    However, with this interface you can also set `axis` equal to the long
    axis of the antenna, in which case `angle` is an azimuth angle.  In this
    manner one can determine where the antenna boresight intersects the ground
    at a given range and therefore determine the Doppler from pointing.

    Arguments:
        radar_xyz   Position of antenna phase center, meters ECEF XYZ.
        axis        Cone generating axis (typically velocity), ECEF XYZ.
        angle       Complement of cone angle, radians.
        range       Range to target, meters.
        dem         Digital elevation model, meters above ellipsoid,
        side        Radar look direction.  Can be string "left" or "right"
                    or pybind11_isce.core.LookSide object.
        threshold   Range convergence threshold, meters, default=0.05.
        maxiter     Maximum iterations, default=50.
        extraiter   Additional iterations, default=50.

    Returns ECEF XYZ of target in meters.
    )");
}
