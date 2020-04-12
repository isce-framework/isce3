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
        .def(py::init<double, isce::core::dataInterpMethod>(),
            py::arg("height") = 0.0,
            py::arg("method") = isce::core::BILINEAR_METHOD)
        // For convenience allow a string, too.
        .def(py::init([](double h, const std::string & method) {
                auto m = parseDataInterpMethod(method);
                return new DI(h, m);
            }),
            py::arg("height") = 0.0,
            py::arg("method") = "bilinear")

        .def("loadDEM",
            py::overload_cast<isce::io::Raster&>(&DI::loadDEM))
        .def("loadDEM",
            py::overload_cast<isce::io::Raster&, double, double, double, double>
                (&DI::loadDEM),
            py::arg("raster"), py::arg("minX"), py::arg("maxX"),
                py::arg("minY"), py::arg("maxY"))

        .def("interpolateLonLat", &DI::interpolateLonLat)
        .def("interpolateXY", &DI::interpolateXY)

        .def_property("refHeight",
            py::overload_cast<>(&DI::refHeight, py::const_),
            py::overload_cast<double>(&DI::refHeight))
        .def_property_readonly("haveRaster", &DI::haveRaster)
        .def_property("interpMethod",
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
        .def_property_readonly("xStart", py::overload_cast<>(&DI::xStart, py::const_))
        .def_property_readonly("yStart", py::overload_cast<>(&DI::yStart, py::const_))
        .def_property_readonly("deltaX", py::overload_cast<>(&DI::deltaX, py::const_))
        .def_property_readonly("deltaY", py::overload_cast<>(&DI::deltaY, py::const_))
        .def_property_readonly("width", py::overload_cast<>(&DI::width, py::const_))
        .def_property_readonly("length", py::overload_cast<>(&DI::length, py::const_))
        .def_property_readonly("epsgCode", py::overload_cast<>(&DI::epsgCode, py::const_))
        ;
}
