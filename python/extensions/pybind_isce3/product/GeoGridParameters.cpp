#include "GeoGridParameters.h"

namespace py = pybind11;

using isce3::product::GeoGridParameters;

void addbinding(pybind11::class_<GeoGridParameters> & pyGeoGridParams)
{
    pyGeoGridParams
        .def(py::init<double, double, double, double, int, int, int>(),
            py::arg("start_x") = 0.0,
            py::arg("start_y") = 0.0,
            py::arg("spacing_x") = 0.0,
            py::arg("spacing_y") = 0.0,
            py::arg("width") = 0,
            py::arg("length") = 0,
            py::arg("epsg") = 4326)
        .def_property("start_x",
            py::overload_cast<>(&GeoGridParameters::startX, py::const_),
            py::overload_cast<double>(&GeoGridParameters::startX))
        .def_property("start_y",
            py::overload_cast<>(&GeoGridParameters::startY, py::const_),
            py::overload_cast<double>(&GeoGridParameters::startY))
        .def_property("spacing_x",
            py::overload_cast<>(&GeoGridParameters::spacingX, py::const_),
            py::overload_cast<double>(&GeoGridParameters::spacingX))
        .def_property("spacing_y",
            py::overload_cast<>(&GeoGridParameters::spacingY, py::const_),
            py::overload_cast<double>(&GeoGridParameters::spacingY))
        .def_property("width",
            py::overload_cast<>(&GeoGridParameters::width, py::const_),
            py::overload_cast<int>(&GeoGridParameters::width))
        .def_property("length",
            py::overload_cast<>(&GeoGridParameters::length, py::const_),
            py::overload_cast<int>(&GeoGridParameters::length))
        .def_property("epsg",
            py::overload_cast<>(&GeoGridParameters::epsg, py::const_),
            py::overload_cast<int>(&GeoGridParameters::epsg))
        ;
}
