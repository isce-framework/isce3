#include "geo2rdr_roots.h"

#include <isce3/core/LookSide.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Vector.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/geometry/geo2rdr_roots.h>
#include <tuple>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind_isce3/core/LookSide.h>

namespace py = pybind11;

using namespace isce3::core;
using namespace isce3::error;

void addbinding_geo2rdr_roots(py::module & m)
{
    m.def("geo2rdr_bracket", [](const Vec3 & x, const Orbit & orbit,
            const LUT2d<double> & doppler, double wvl, py::object pySide,
            double tol_aztime, std::optional<double> time_start,
            std::optional<double> time_end) {
        double aztime, range;
        const auto side = duck_look_side(pySide);
        int success = isce3::geometry::geo2rdr_bracket(
                x, orbit, doppler, aztime, range, wvl, side, tol_aztime,
                time_start, time_end);
        if (!success) {
            throw std::runtime_error("failed to converge");
        }
        return std::make_tuple(aztime, range);
    },
    py::arg("xyz"), py::arg("orbit"), py::arg("doppler"), py::arg("wavelength"),
    py::arg("side"),
    py::arg("tol_aztime") = isce3::geometry::detail::DEFAULT_TOL_AZ_TIME,
    py::arg("time_start") = std::nullopt, py::arg("time_end") = std::nullopt,
    R"(
    Convert geographic coordinates to radar coordinates.

    Parameters
    ----------
    xyz : array_like
        Target position, ECEF XYZ in meters
    orbit : isce3.core.Orbit
        Reference orbit defining the radar coordinate system.
    doppler : isce3.core.LUT2d
        Reference Doppler centroid in Hz as function of (time, range) that
        defines the radar coordinate system.  Note that this isn't necessarily
        the processed Doppler centroid.  For example, NISAR products are
        generated on a zero-Doppler grid where doppler(t,r) = 0.
    wavelength : float
        Wavelength in meters corresponding to the provided Doppler.  If
        doppler(t,r)=0 then wavelength has no effect on the solution as long as
        it is finite and nonzero.
    side : Union[isce3.core.LookSide, str]
        Radar look direction, LookSide.Left or LookSide.Right
        or "left" or "right"
    tol_aztime : float, optional
        Azimuth time convergence tolerance in seconds.
    time_start : float, optional
        Start of search interval, in seconds
        Defaults to orbit.start_time
    time_end : float, optional
        End of search interval, in seconds
        Defaults to orbit.end_time

    Returns
    -------
    aztime : float
        Time when Doppler centroid crosses target, in seconds since
        orbit.reference_epoch
    range : float
        Distance to target (in meters) at aztime

    Notes
    -----
    By default, the search region is initialized with the start and end times of
    the orbit.  This means the algorithm may fail or return nonsense if too much
    orbit data are provided, e.g., a 30 hour time series containing several
    revolutions around the Earth.  One should trim the orbit data to a
    reasonable interval before calling this routine (e.g., with the
    isce3.core.Orbit.crop method) or provide a custom bracket (time_start,
    time_end).

    This algorithm assumes there is a unique time when the target Doppler equals
    the reference Doppler.  This is typically true for stripmap SAR geometries
    but may fail for more exotic scenarios like circular SAR, a point at the
    center of the earth, or highly irregular trajectories.
    )");
}
