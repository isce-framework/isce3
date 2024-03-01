#include "RadarGridParameters.h"

#include <stdexcept>
#include <string>

#include <isce3/core/DateTime.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/Linspace.h>
#include <isce3/product/RadarGridProduct.h>

namespace py = pybind11;

using isce3::product::RadarGridParameters;
using isce3::core::DateTime;
using isce3::core::LookSide;

void addbinding(pybind11::class_<RadarGridParameters> & pyRadarGridParameters)
{
    pyRadarGridParameters
        .def(py::init([](const std::string &h5file, const char freq)
                {
                    // open file
                    isce3::io::IH5File file(h5file);

                    // instantiate and load a product
                    isce3::product::RadarGridProduct product(file);

                    // return swath from product
                    return RadarGridParameters(product, freq);
                }),
                py::arg("h5file"), py::arg("freq")='A')
        .def(py::init<double, double, double, double, double, LookSide,
                size_t, size_t, DateTime>(),
                py::arg("sensing_start"),
                py::arg("wavelength"),
                py::arg("prf"),
                py::arg("starting_range"),
                py::arg("range_pixel_spacing"),
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

                    LookSide side = isce3::core::parseLookSide(look_side);

                    return RadarGridParameters(sensing_start, wavelength, prf,
                            starting_range, range_pixel_spacing, side, length,
                            width, ref_epoch);
                }),
                py::arg("sensing_start"),
                py::arg("wavelength"),
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
        .def_property_readonly("sensing_mid", &RadarGridParameters::sensingMid)
        .def_property_readonly("sensing_stop", &RadarGridParameters::sensingStop)
        .def("sensing_time", &RadarGridParameters::sensingTime,
                py::arg("line"))
        .def_property_readonly("sensing_times", [](const RadarGridParameters& self) {
                return isce3::core::Linspace(self.sensingStart(),
                        1.0 / self.prf(), self.length());
        })
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
        .def("offset_and_resize", &RadarGridParameters::offsetAndResize,
                py::arg("yoff"), py::arg("xoff"), py::arg("ysize"), py::arg("xsize"))
        .def("multilook", &RadarGridParameters::multilook,
                py::arg("azlooks"), py::arg("rglooks"))
        .def("slant_range", &RadarGridParameters::slantRange,
                py::arg("sample"))
        .def("slant_range_index", &RadarGridParameters::slantRangeIndex,
                py::arg("slant_range"))
        .def_property_readonly("slant_ranges", [](const RadarGridParameters& self) {
                return isce3::core::Linspace(self.startingRange(),
                        self.rangePixelSpacing(), self.width());
        })
        .def_property_readonly("sensing_times", [](const RadarGridParameters& self) {
                return isce3::core::Linspace(self.sensingStart(),
                        1.0 / self.prf(), self.length());
        })
        .def("azimuth_index", &RadarGridParameters::azimuthIndex,
                py::arg("az_time"))
        // slice to get subset of RGP
        .def("__getitem__", [](const RadarGridParameters& self, py::tuple key) {
                if (key.size() != 2) {
                        throw std::invalid_argument("require 2 slices");
                }
                auto islice = key[0].cast<py::slice>();
                auto jslice = key[1].cast<py::slice>();
                py::ssize_t start, stop, step, slicelen;

                if (!islice.compute(self.length(), &start, &stop, &step, &slicelen))
                        throw std::invalid_argument("bad row slice");
                if (step <= 0)
                        throw py::index_error("cannot reverse grid");
                double prf = self.prf() / step;
                double t0 = self.sensingStart() + start / self.prf();
                auto nt = slicelen;

                if (!jslice.compute(self.width(), &start, &stop, &step, &slicelen))
                        throw std::invalid_argument("bad column slice");
                if (step <= 0)
                        throw py::index_error("cannot reverse grid");
                double dr = self.rangePixelSpacing() * step;
                double r0 = self.startingRange() + start * self.rangePixelSpacing();
                auto nr = slicelen;

                return RadarGridParameters(t0, self.wavelength(), prf, r0, dr,
                        self.lookSide(), nt, nr, self.refEpoch());
        })
        .def("copy", [](const RadarGridParameters& self) {
                return RadarGridParameters(self);
        })
        // resize the radar grid with with the start and stop points kept
        .def("resize_and_keep_startstop", &RadarGridParameters::resizeKeepStartStop,
                py::arg("ysize"),
                py::arg("xsize"), R"(
        Resize the RadarGridParameters object by using the ysize and xsize, and
        the start and stop points are kept.

        Parameters
        ----------
        ysize : int
          The number of samples along the azimuth direction. Must be >1.
        xsize : int
          The number of samples along the slant range direction. Must be >1.

        Returns
        -------
        RadarGridParameters
          The resized radar grid.
            )")
        // add margins to the radar grid
        .def("add_margin", &RadarGridParameters::addMargin,
                py::arg("ymargin"),
                py::arg("xmargin"),
                py::arg("side") = "both",  R"(
        Add margins to the RadarGridParameters object

        Parameters
        ----------
        ymargin : int
          The number of samples along the azimuth. Must be >0.
        xmargin : int
          The number of samples along the slant range. Must be >0.
        side : str
          The side where the margin will be added, and
          options are 'start', 'stop', and 'both'. Defaults to 'both'.

        Returns
        -------
        RadarGridParameters
          The radar grid with the margins added.
            )")
        .def_property_readonly("shape", [](const RadarGridParameters& self) {
                auto shape = py::tuple(2);
                shape[0] = self.length();
                shape[1] = self.width();
                return shape;
        })
        // FIXME Attribute names don't match ctor names.
        .def("__str__", [](const py::object self) {
                std::vector<std::string> keys {"sensing_start", "wavelength",
                        "prf", "starting_range", "range_pixel_spacing",
                        "lookside", "length", "width", "ref_epoch"};
                std::string out("RadarGridParameters(");
                for (auto it = keys.begin(); it != keys.end(); ++it) {
                        auto key = *it;
                        auto ckey = key.c_str();
                        out += key + "=" + std::string(py::str(self.attr(ckey)));
                        if (it != keys.end() - 1)
                                out += ", ";
                }
                return out + ")";
        })
        .def("sensing_datetime", &RadarGridParameters::sensingDateTime,
                py::arg("line") = 0)
        .def("contains", &RadarGridParameters::contains,
                py::arg("azimuth_time"),
                py::arg("slant_range"))
        ;
}
