#include "geo2rdr.h"

#include <string>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/io/Raster.h>
#include <isce3/product/RadarGridParameters.h>

using isce3::geometry::Geo2rdr;

namespace py = pybind11;

void addbinding(py::class_<Geo2rdr> & pyGeo2Rdr)
{
    pyGeo2Rdr
        .def(py::init([](const isce3::product::RadarGridParameters & radar_grid,
             const isce3::core::Orbit & orbit,
             const isce3::core::Ellipsoid & ellipsoid,
             const isce3::core::LUT2d<double> & doppler,
             const double threshold,
             const int numiter)
            {
                auto geo2rdr_obj = Geo2rdr(radar_grid, orbit, ellipsoid, doppler);
                geo2rdr_obj.threshold(threshold);
                geo2rdr_obj.numiter(numiter);
                return geo2rdr_obj;
            }),
            py::arg("radar_grid"),
            py::arg("orbit"),
            py::arg("ellipsoid"),
            py::arg("doppler") = isce3::core::LUT2d<double>(),
            py::arg("threshold") = 0.05,
            py::arg("numiter") = 25)
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
        ;
}
