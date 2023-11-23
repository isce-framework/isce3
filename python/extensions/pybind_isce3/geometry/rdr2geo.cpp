#include "rdr2geo.h"

#include <cmath>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind_isce3/core/LookSide.h>
#include <stdexcept>
#include <string>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Vector.h>
#include <isce3/focus/Backproject.h> // TODO better place for defaults
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/detail/Rdr2Geo.h>
#include <isce3/geometry/geometry.h>
#include <isce3/io/Raster.h>
#include <isce3/product/RadarGridParameters.h>

using isce3::geometry::DEMInterpolator;
using isce3::geometry::Topo;
using isce3::geometry::detail::Rdr2GeoParams;

namespace py = pybind11;

void addbinding(py::class_<Rdr2GeoParams>& pyRdr2GeoParams)
{
    pyRdr2GeoParams
            .def(py::init<const double, const int, const int>(),
                    py::arg("threshold") = 0.05, py::arg("maxiter") = 25,
                    py::arg("extraiter") = 10)
            .def_readwrite("threshold", &Rdr2GeoParams::threshold)
            .def_readwrite("maxiter", &Rdr2GeoParams::maxiter)
            .def_readwrite("extraiter", &Rdr2GeoParams::extraiter);
}

static Rdr2GeoParams handle_r2g_kwargs(py::kwargs kw)
{
    Rdr2GeoParams params;
    if (kw.contains("threshold")) {
        params.threshold = kw["threshold"].cast<double>();
    }
    if (kw.contains("maxiter")) {
        params.maxiter = kw["maxiter"].cast<int>();
    }
    if (kw.contains("extraiter")) {
        params.extraiter = kw["extraiter"].cast<int>();
    }
    return params;
}

void addbinding_rdr2geo(py::module& m)
{
    // Use lambdas to duck type look side (str or LookSide) and raise
    // exceptions on error.
    Rdr2GeoParams defaults;
    m.def(
             "rdr2geo_cone", // name matches old Cython version
             [](const Vec3& radarXYZ, const Vec3& axis, double angle,
                     double range, const DEMInterpolator& dem,
                     py::object pySide, py::kwargs r2g_kw) {
                 // duck type side
                 auto side = duck_look_side(pySide);
                 auto opt = handle_r2g_kwargs(r2g_kw);
                 Vec3 targetXYZ;
                 int converged = rdr2geo(radarXYZ, axis, angle, range, dem,
                         targetXYZ, side, opt.threshold, opt.maxiter,
                         opt.extraiter);
                 if (!converged)
                     throw std::runtime_error("rdr2geo failed to converge");
                 return targetXYZ;
             },
             py::arg("radar_xyz"), py::arg("axis"), py::arg("angle"),
             py::arg("range"), py::arg("dem"), py::arg("side"),
             R"(
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
    )")

            // interface close to old isce3.geometry.rdr2geo_point
            .def(
                    "rdr2geo",
                    [](double aztime, double range, const Orbit& orbit,
                            py::object pySide, double doppler,
                            double wavelength, const DEMInterpolator& dem,
                            const Ellipsoid& ellipsoid, py::kwargs r2g_kw) {
                        if ((std::abs(doppler) > 0.0) && (wavelength <= 0.0)) {
                            throw std::invalid_argument(
                                    "need valid wavelength when doppler is "
                                    "nonzero, "
                                    "got wavelength=" +
                                    std::to_string(doppler));
                        }
                        // duck type side
                        auto side = duck_look_side(pySide);
                        auto opt = handle_r2g_kwargs(r2g_kw);
                        auto midx = dem.midX();
                        // FIXME figure out dem.midLonLat() segfaults
                        Vec3 targetLLH {
                                dem.midX(), dem.midY(), dem.refHeight()};
                        int converged = rdr2geo(aztime, range, doppler, orbit,
                                ellipsoid, dem, targetLLH, wavelength, side,
                                opt.threshold, opt.maxiter, opt.extraiter);
                        if (!converged)
                            throw std::runtime_error(
                                    "rdr2geo failed to converge");
                        return targetLLH;
                    },
                    py::arg("aztime"), py::arg("range"), py::arg("orbit"),
                    py::arg("side"), py::arg("doppler") = 0.0,
                    py::arg("wavelength") = 0.0,
                    py::arg("dem") = DEMInterpolator(),
                    py::arg("ellipsoid") = Ellipsoid(), R"(
    Radar geometry coordinates to map coordinates transformer

    Arguments:
        aztime      Azimuth time of Doppler intersection,
                    in seconds since orbit.reference_epoch
        range       Range to target, meters.
        orbit       isce3.core.Orbit object.
        side        Radar look direction.  Can be string "left" or "right"
                    or isce3.core.LookSide object.
        doppler     Doppler that defines geometry, Hz.  (default=0)
        wavelength  Wavelength corresponding to Doppler measurement, m.
                    Must be nonzero if Doppler is nonzero.
        dem         Digital elevation model, meters above ellipsoid.
        ellipsoid   Ellipsoid for computing LLH, default=WGS84.
        threshold   Range convergence threshold, meters.
        maxiter     Maximum iterations.
        extraiter   Additional iterations.

    Returns (longitude, latitude, height) of target in (rad,rad,m) units.
        )");
}

void addbinding(py::class_<Topo>& pyRdr2Geo)
{
    pyRdr2Geo
            .def(py::init([](const isce3::product::RadarGridParameters&
                                          radar_grid,
                                  const isce3::core::Orbit& orbit,
                                  const isce3::core::Ellipsoid& ellipsoid,
                                  const isce3::core::LUT2d<double>& doppler,
                                  const double threshold, const int numiter,
                                  const int extraiter,
                                  const dataInterpMethod dem_interp_method,
                                  const int epsg_out, const bool compute_mask,
                                  const int lines_per_block) {
                auto rdr2geo_obj = Topo(radar_grid, orbit, ellipsoid, doppler);
                rdr2geo_obj.threshold(threshold);
                rdr2geo_obj.numiter(numiter);
                rdr2geo_obj.extraiter(extraiter);
                rdr2geo_obj.demMethod(dem_interp_method);
                rdr2geo_obj.epsgOut(epsg_out);
                rdr2geo_obj.computeMask(compute_mask);
                rdr2geo_obj.linesPerBlock(lines_per_block);
                return rdr2geo_obj;
            }),
                    py::arg("radar_grid"), py::arg("orbit"),
                    py::arg("ellipsoid"),
                    py::arg("doppler") = isce3::core::LUT2d<double>(),
                    py::arg("threshold") = 0.05, py::arg("numiter") = 25,
                    py::arg("extraiter") = 10,
                    py::arg("dem_interp_method") =
                            isce3::core::BIQUINTIC_METHOD,
                    py::arg("epsg_out") = 4326, py::arg("compute_mask") = true,
                    py::arg("lines_per_block") = 1000)
            .def("topo",
                    py::overload_cast<isce3::io::Raster&, const std::string&>(
                            &Topo::topo),
                    py::arg("dem_raster"), py::arg("outdir"))
            .def("topo",
                    py::overload_cast<isce3::io::Raster&, isce3::io::Raster*,
                            isce3::io::Raster*, isce3::io::Raster*,
                            isce3::io::Raster*, isce3::io::Raster*,
                            isce3::io::Raster*, isce3::io::Raster*,
                            isce3::io::Raster*, isce3::io::Raster*,
                            isce3::io::Raster*, isce3::io::Raster*>(
                            &Topo::topo),
                    py::arg("dem_raster"), py::arg("x_raster") = nullptr,
                    py::arg("y_raster") = nullptr,
                    py::arg("height_raster") = nullptr,
                    py::arg("incidence_angle_raster") = nullptr,
                    py::arg("heading_angle_raster") = nullptr,
                    py::arg("local_incidence_angle_raster") = nullptr,
                    py::arg("local_psi_raster") = nullptr,
                    py::arg("simulated_amplitude_raster") = nullptr,
                    py::arg("layover_shadow_raster") = nullptr,
                    py::arg("ground_to_sat_east_raster") = nullptr,
                    py::arg("ground_to_sat_north_raster") = nullptr,
                    R"(
        Run topo and write to user created rasters

        Parameters
        ----------
        dem_raster: isce3.io.Raster
            Input DEM raster
        x_raster: isce3.io.Raster
            Output raster for X coordinate in requested projection system
            (meters or degrees)
        y_raster: isce3.io.Raster
            Output raster for Y cooordinate in requested projection system
            (meters or degrees)
        height_raster: isce3.io.Raster
            Output raster for height above ellipsoid (meters)
        incidence_raster: isce3.io.Raster
            Output raster for incidence angle (degrees) computed from vertical
            at target
        heading_angle_raster: isce3.io.Raster
            Output raster for azimuth angle (degrees) computed anti-clockwise
            from EAST (Right hand rule)
        local_incidence_raster: isce3.io.Raster
            Output raster for local incidence angle (degrees) at target
        local_psi_raster: isce3.io.Raster
            Output raster for local projection angle (degrees) at target
        simulated_amplitude_raster: isce3.io.Raster
            Output raster for simulated amplitude image.
        layover_shadow_raster: isce3.io.Raster
            Output raster for layover/shadow mask.
        ground_to_sat_east_raster: isce3.io.Raster
            Output raster for east component of ground to satellite unit vector
        ground_to_sat_north_raster: isce3.io.Raster
            Output raster for north component of ground to satellite unit vector
                    )")
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
            .def_property("lines_per_block",
                    py::overload_cast<>(&Topo::linesPerBlock, py::const_),
                    py::overload_cast<size_t>(&Topo::linesPerBlock));
}
