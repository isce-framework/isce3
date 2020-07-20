#include "rdr2geo.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Vector.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/geometry.h>
#include <isce3/io/Raster.h>
#include <isce3/product/RadarGridParameters.h>

using isce::geometry::DEMInterpolator;
using isce::geometry::Topo;

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
        threshold   Range convergence threshold, meters.
        maxiter     Maximum iterations.
        extraiter   Additional iterations.

    Returns ECEF XYZ of target in meters.
    )");
}

void addbinding(py::class_<Topo> & pyRdr2Geo)
{
    pyRdr2Geo
        .def(py::init([](const isce::product::RadarGridParameters & radar_grid,
             const isce::core::Orbit & orbit,
             const isce::core::Ellipsoid & ellipsoid,
             const isce::core::LUT2d<double> & doppler,
             const double threshold,
             const int numiter,
             const int extraiter,
             const dataInterpMethod dem_interp_method,
             const int epsg_out,
             const bool compute_mask)
            {
                auto rdr2geo_obj = Topo(radar_grid, orbit, ellipsoid, doppler);
                rdr2geo_obj.threshold(threshold);
                rdr2geo_obj.numiter(numiter);
                rdr2geo_obj.extraiter(extraiter);
                rdr2geo_obj.demMethod(dem_interp_method);
                rdr2geo_obj.epsgOut(epsg_out);
                rdr2geo_obj.computeMask(compute_mask);
                return rdr2geo_obj;
            }),
            py::arg("radar_grid"),
            py::arg("orbit"),
            py::arg("ellipsoid"),
            py::arg("doppler") = {},
            py::arg("threshold") = 0.05,
            py::arg("numiter") = 25,
            py::arg("extraiter") = 10,
            py::arg("dem_interp_method") = isce::core::BIQUINTIC_METHOD,
            py::arg("epsg_out") = 4326,
            py::arg("compute_mask") = false)
        .def("topo", py::overload_cast<isce::io::Raster &, const std::string &>
                (&Topo::topo),
                py::arg("dem_raster"),
                py::arg("outdir"))
        .def_property_readonly("orbit", &Topo::orbit)
        .def_property_readonly("ellipsoid", &Topo::ellipsoid)
        .def_property_readonly("doppler", &Topo::doppler)
        .def_property_readonly("radar_grid", &Topo::radarGridParameters)
        .def_property("threshold",
                py::overload_cast<>(&Topo::threshold, py::const_),
                py::overload_cast<double>(&Topo::threshold))
        .def_property("numiter",
                py::overload_cast<>(&Topo::numiter, py::const_),
                py::overload_cast<int>(&Topo::numiter))
        .def_property("extraiter",
                py::overload_cast<>(&Topo::extraiter, py::const_),
                py::overload_cast<int>(&Topo::extraiter))
        .def_property("dem_interp_method",
                py::overload_cast<>(&Topo::demMethod, py::const_),
                py::overload_cast<dataInterpMethod>(&Topo::demMethod))
        .def_property("epsg_out",
                py::overload_cast<>(&Topo::epsgOut, py::const_),
                py::overload_cast<int>(&Topo::epsgOut))
        .def_property("compute_mask",
                py::overload_cast<>(&Topo::computeMask, py::const_),
                py::overload_cast<bool>(&Topo::computeMask))
        ;
}
