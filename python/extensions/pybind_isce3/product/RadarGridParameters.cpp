#include "RadarGridParameters.h"

#include <string>

#include <isce/core/DateTime.h>
#include <isce/core/LookSide.h>

namespace py = pybind11;

using isce::product::RadarGridParameters;
using isce::core::DateTime;
using isce::core::LookSide;

void addbinding(pybind11::class_<RadarGridParameters> & pyRadarGridParameters)
{
    pyRadarGridParameters
        .def(py::init([](const std::string &h5file, const char freq)
                {
                    // open file
                    isce::io::IH5File file(h5file);

                    // instantiate and load a product
                    isce::product::Product product(file);

                    // return swath from product
                    return RadarGridParameters(product, freq);
                }),
                py::arg("h5file"), py::arg("freq")='A')
        .def(py::init<double, double, double, double, double, LookSide,
                size_t, size_t, DateTime>(),
                py::arg("sensing_start"),
                py::arg("wavelngth"),
                py::arg("prf"),
                py::arg("starting_range"),
                py::arg("range_pxl_spacing"),
                py::arg("lookside"),
                py::arg("length"),
                py::arg("width"),
                py::arg("ref_epoch"))
        .def(py::init([](double sensing_start,
                         double wavelength,
                         double prf,
                         double starting_range,
                         double range_pixel_spacing,
                         const std::string& look_side,
                         size_t length,
                         size_t width,
                         const DateTime& ref_epoch) {

                    LookSide side = isce::core::parseLookSide(look_side);

                    return RadarGridParameters(sensing_start, wavelength, prf,
                            starting_range, range_pixel_spacing, side, length,
                            width, ref_epoch);
                }),
                py::arg("sensing_start"),
                py::arg("wavelngth"),
                py::arg("prf"),
                py::arg("starting_range"),
                py::arg("range_pixel_spacing"),
                py::arg("look_side"),
                py::arg("length"),
                py::arg("width"),
                py::arg("ref_epoch"))
        .def_property_readonly("size", &RadarGridParameters::size)
        .def_property_readonly("end_range", &RadarGridParameters::endingRange)
        .def_property_readonly("mid_range", &RadarGridParameters::midRange)
        .def_property_readonly("az_time_interval",  &RadarGridParameters::azimuthTimeInterval)
        .def_property("lookside",
                 py::overload_cast<>(&RadarGridParameters::lookSide, py::const_),
                 py::overload_cast<LookSide>(&RadarGridParameters::lookSide))
        .def_property("sensing_start",
                py::overload_cast<>(&RadarGridParameters::sensingStart, py::const_),
                py::overload_cast<const double&>(&RadarGridParameters::sensingStart))
        .def_property("ref_epoch",
                py::overload_cast<>(&RadarGridParameters::refEpoch, py::const_),
                py::overload_cast<const DateTime &>(&RadarGridParameters::refEpoch))
        .def_property("wavelength",
                py::overload_cast<>(&RadarGridParameters::wavelength, py::const_),
                py::overload_cast<const double&>(&RadarGridParameters::wavelength))
        .def_property("prf",
                py::overload_cast<>(&RadarGridParameters::prf, py::const_),
                py::overload_cast<const double&>(&RadarGridParameters::prf))
        .def_property("starting_range",
                py::overload_cast<>(&RadarGridParameters::startingRange, py::const_),
                py::overload_cast<const double&>(&RadarGridParameters::startingRange))
        .def_property("range_pixel_spacing",
                py::overload_cast<>(&RadarGridParameters::rangePixelSpacing, py::const_),
                py::overload_cast<const double&>(&RadarGridParameters::rangePixelSpacing))
        .def_property("width",
                py::overload_cast<>(&RadarGridParameters::width, py::const_),
                py::overload_cast<const size_t&>(&RadarGridParameters::width))
        .def_property("length",
                py::overload_cast<>(&RadarGridParameters::length, py::const_),
                py::overload_cast<const size_t&>(&RadarGridParameters::length))
        .def("multilook", &RadarGridParameters::multilook,
                py::arg("azlooks"), py::arg("rglooks"))
        .def("slant_range", &RadarGridParameters::slantRange,
                py::arg("sample"))
        ;
}
