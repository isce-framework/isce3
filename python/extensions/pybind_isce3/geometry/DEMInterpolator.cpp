#include "DEMInterpolator.h"

#include <isce/core/Constants.h>
#include <isce/io/Raster.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <string>

namespace py = pybind11;

using DI = isce::geometry::DEMInterpolator;

void addbinding(pybind11::class_<DI> & pyDEMInterpolator)
{
    pyDEMInterpolator
        .def(py::init<double, isce::core::dataInterpMethod, int>(),
            py::arg("height") = 0.0,
            py::arg("method") = isce::core::BILINEAR_METHOD,
            py::arg("epsg") = 4326)
        // For convenience allow a string, too.
        .def(py::init([](double h, const std::string & method, int epsg) {
                auto m = parseDataInterpMethod(method);
                return new DI(h, m, epsg);
            }),
            py::arg("height") = 0.0,
            py::arg("method") = "bilinear",
            py::arg("epsg") = 4326)

        .def("load_dem",
            py::overload_cast<isce::io::Raster&>(&DI::loadDEM))
        .def("load_dem",
            py::overload_cast<isce::io::Raster&, double, double, double, double>
                (&DI::loadDEM),
            py::arg("raster"), py::arg("min_x"), py::arg("max_x"),
                py::arg("min_y"), py::arg("max_y"))

        .def("interpolate_lonlat", &DI::interpolateLonLat)
        .def("interpolate_xy", &DI::interpolateXY)

        .def_property("ref_height",
            py::overload_cast<>(&DI::refHeight, py::const_),
            py::overload_cast<double>(&DI::refHeight))
        .def_property_readonly("have_raster", &DI::haveRaster)
        .def_property("interp_method",
            py::overload_cast<>(&DI::interpMethod, py::const_),
            py::overload_cast<isce::core::dataInterpMethod>(&DI::interpMethod))

        // Define all these as readonly even though writable in C++ API.
        // Probably better to just convert your data to a GDAL format than try
        // to build a DEM on the fly.
        .def_property_readonly("data", [](DI & self) { // .data() isn't const
            if (!self.haveRaster()) {
                throw std::out_of_range("Tried to access DEM data but size=0");
            }
            using namespace Eigen;
            using MatF = Eigen::Matrix<float, Dynamic, Dynamic, RowMajor>;
            Map<const MatF> mat(self.data(), self.length(), self.width());
            return mat;
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("x_start", py::overload_cast<>(&DI::xStart, py::const_))
        .def_property_readonly("y_start", py::overload_cast<>(&DI::yStart, py::const_))
        .def_property_readonly("delta_x", py::overload_cast<>(&DI::deltaX, py::const_))
        .def_property_readonly("delta_y", py::overload_cast<>(&DI::deltaY, py::const_))
        .def_property_readonly("width", py::overload_cast<>(&DI::width, py::const_))
        .def_property_readonly("length", py::overload_cast<>(&DI::length, py::const_))
        .def_property_readonly("epsg_code", py::overload_cast<>(&DI::epsgCode, py::const_))
        ;
}
