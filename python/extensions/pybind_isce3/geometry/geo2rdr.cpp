#include "geo2rdr.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind_isce3/core/LookSide.h>
#include <stdexcept>
#include <string>
#include <utility>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Vector.h>
#include <isce3/focus/Backproject.h>
#include <isce3/geometry/geometry.h>
#include <isce3/io/Raster.h>
#include <isce3/product/RadarGridParameters.h>

using isce3::geometry::detail::Geo2RdrParams;
using isce3::geometry::Geo2rdr;
using isce3::geometry::geo2rdr;

namespace py = pybind11;

void addbinding(py::class_<Geo2RdrParams> &pyGeo2RdrParams)
{
    pyGeo2RdrParams
        .def(py::init<const double, const int, const double>(),
            py::arg("threshold") = 1e-8,
            py::arg("maxiter") = 50,
            py::arg("delta_range") = 10)
        .def_readwrite("threshold", &Geo2RdrParams::threshold)
        .def_readwrite("maxiter", &Geo2RdrParams::maxiter)
        .def_readwrite("delta_range", &Geo2RdrParams::delta_range)
        ;
}

void addbinding(py::class_<Geo2rdr> & pyGeo2Rdr)
{
    const isce3::geometry::detail::Geo2RdrParams defaults;
    pyGeo2Rdr
        .def(py::init([](const isce3::product::RadarGridParameters & radar_grid,
             const isce3::core::Orbit & orbit,
             const isce3::core::Ellipsoid & ellipsoid,
             const isce3::core::LUT2d<double> & doppler,
             const double threshold,
             const int numiter,
             const int lines_per_block)
            {
                auto geo2rdr_obj = Geo2rdr(radar_grid, orbit, ellipsoid, doppler);
                geo2rdr_obj.threshold(threshold);
                geo2rdr_obj.numiter(numiter);
                geo2rdr_obj.linesPerBlock(lines_per_block);
                return geo2rdr_obj;
            }),
            py::arg("radar_grid"),
            py::arg("orbit"),
            py::arg("ellipsoid"),
            py::arg("doppler") = isce3::core::LUT2d<double>(),
            py::arg("threshold") = defaults.threshold,
            py::arg("numiter") = defaults.maxiter,
            py::arg("lines_per_block") = 1000)
        .def("geo2rdr", py::overload_cast<isce3::io::Raster &, const std::string &,
                double, double>
                (&Geo2rdr::geo2rdr),
                py::arg("dem_raster"),
                py::arg("outdir"),
                py::arg("az_shift") = 0.0,
                py::arg("rg_shift") = 0.0)
        .def_property_readonly("orbit", &Geo2rdr::orbit)
        .def_property_readonly("ellipsoid", &Geo2rdr::ellipsoid)
        .def_property_readonly("doppler", &Geo2rdr::doppler)
        .def_property_readonly("radar_grid", &Geo2rdr::radarGridParameters)
        .def_property("threshold",
                py::overload_cast<>(&Geo2rdr::threshold, py::const_),
                py::overload_cast<double>(&Geo2rdr::threshold))
        .def_property("numiter",
                py::overload_cast<>(&Geo2rdr::numiter, py::const_),
                py::overload_cast<int>(&Geo2rdr::numiter))
        .def_property("lines_per_block",
                py::overload_cast<>(&Geo2rdr::linesPerBlock, py::const_),
                py::overload_cast<size_t>(&Geo2rdr::linesPerBlock))
        ;
}

void addbinding_geo2rdr(pybind11::module& m)
{
    const isce3::geometry::detail::Geo2RdrParams defaults;
    m.def("geo2rdr",
        [](const Vec3& lon_lat_height, const Ellipsoid& ellipsoid, const Orbit& orbit,
            const LUT2d<double>& doppler, double wavelength, py::object py_side,
            double threshold, int maxiter, double delta_range) {
                auto side = duck_look_side(py_side);
                double aztime, slant_range;
                int converged = geo2rdr(
                        lon_lat_height, ellipsoid, orbit, doppler,
                        aztime, slant_range,
                        wavelength, side,
                        threshold, maxiter, delta_range);
                if (!converged)
                    throw std::runtime_error("geo2rdr failed to converge");
                return std::make_pair(aztime, slant_range);
        },
        py::arg("lon_lat_height"),
        py::arg("ellipsoid")=Ellipsoid(),
        py::arg("orbit"),
        py::arg("doppler"),
        py::arg("wavelength"),
        py::arg("side"),
        py::arg("threshold")=defaults.threshold,
        py::arg("maxiter")=defaults.maxiter,
        py::arg("delta_range")=defaults.delta_range,
        R"(
    This is the elementary transformation from map geometry to radar geometry.
    The transformation is applicable for a single lon/lat/h coordinate (i.e., a
    single point target).

    Arguments:
        lon_lat_height  Lon/Lat/Hae of target of interest
        ellipsoid       Ellipsoid object
        orbit           Orbit object
        doppler         LUT2d Doppler model
        wavelength      Radar wavelength
        side            Left or Right
        threshold       azimuth time convergence threshold in meters
        maxiter         Maximum number of Newton-Raphson iterations
        delta_range     Step size used for computing derivative of doppler

    Returns:
        aztime          azimuth time of input Lon/Lat/Hae w.r.t reference
                        epoch of the orbit
        slantRange      slant range to input Lon/Lat/Hae
        )"
        );
}
