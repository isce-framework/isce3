#include "geo2rdr.h"

#include <string>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/geometry.h>
#include <isce3/io/Raster.h>
#include <isce3/product/RadarGridParameters.h>

using isce3::cuda::geometry::Geo2rdr;

namespace py = pybind11;

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
                py::arg("rdr2geo_raster"),
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
