#include "Grid.h"

#include <string>

#include <pybind11/stl.h>

#include <isce3/io/IH5.h>
#include <isce3/product/GeoGridProduct.h>
#include <isce3/product/GeoGridParameters.h>

namespace py = pybind11;

using isce3::product::Grid;

void addbinding(pybind11::class_<Grid> & pyGrid)
{
    pyGrid
        .def(py::init<>())
        .def(py::init([](const std::string &h5file, const char freq)
                {
                    // open file
                    isce3::io::IH5File file(h5file);

                    // instantiate and load a product
                    isce3::product::GeoGridProduct product(file);

                    // return grid from product
                    return product.grid(freq);
                }),
                py::arg("h5file"), py::arg("freq"))

        .def_property_readonly("wavelength", &Grid::wavelength)
        .def_property("geogrid",
                py::overload_cast<>(&Grid::geogrid),
                py::overload_cast<isce3::product::GeoGridParameters>(&Grid::geogrid))
        .def_property("range_bandwidth",
                py::overload_cast<>(&Grid::rangeBandwidth, py::const_),
                py::overload_cast<double>(&Grid::rangeBandwidth))
        .def_property("azimuth_bandwidth",
                py::overload_cast<>(&Grid::azimuthBandwidth, py::const_),
                py::overload_cast<double>(&Grid::azimuthBandwidth))
        .def_property("center_frequency",
                py::overload_cast<>(&Grid::centerFrequency, py::const_),
                py::overload_cast<double>(&Grid::centerFrequency))
        .def_property("slant_range_spacing",
                py::overload_cast<>(&Grid::slantRangeSpacing, py::const_),
                py::overload_cast<double>(&Grid::slantRangeSpacing))
        .def_property("zero_doppler_time_spacing",
                py::overload_cast<>(&Grid::zeroDopplerTimeSpacing, py::const_),
                py::overload_cast<double>(&Grid::zeroDopplerTimeSpacing))

        .def_property("start_x",
                py::overload_cast<>(&Grid::startX, py::const_),
                py::overload_cast<double>(&Grid::startX))
        .def_property("start_y",
                py::overload_cast<>(&Grid::startY, py::const_),
                py::overload_cast<double>(&Grid::startY))
        .def_property("spacing_x",
                py::overload_cast<>(&Grid::spacingX, py::const_),
                py::overload_cast<double>(&Grid::spacingX))
        .def_property("spacing_y",
                py::overload_cast<>(&Grid::spacingY, py::const_),
                py::overload_cast<double>(&Grid::spacingY))
        .def_property("width",
                py::overload_cast<>(&Grid::width, py::const_),
                py::overload_cast<int>(&Grid::width))
        .def_property("length",
                py::overload_cast<>(&Grid::length, py::const_),
                py::overload_cast<int>(&Grid::length))
        .def_property("epsg",
                py::overload_cast<>(&Grid::epsg, py::const_),
                py::overload_cast<int>(&Grid::epsg));
}
