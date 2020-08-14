#include "ResampSlc.h"

#include <isce3/core/Constants.h>
#include <isce3/core/LUT2d.h>
#include <isce3/io/Raster.h>

using isce3::image::ResampSlc;

namespace py = pybind11;

void addbinding(py::class_<ResampSlc> & pyResampSlc)
{
    pyResampSlc
        .def(py::init<const isce3::core::LUT2d<double> &, double, double,
                double, double, double>(),
                py::arg("doppler"),
                py::arg("start_range"),
                py::arg("range_pixel_spacing"),
                py::arg("sensing_start"),
                py::arg("prf"),
                py::arg("wavelength"))
        .def(py::init<const isce3::core::LUT2d<double> &, double, double,
                double, double, double, double, double, double>(),
                py::arg("doppler"),
                py::arg("start_range"),
                py::arg("range_pixel_spacing"),
                py::arg("sensing_start"),
                py::arg("prf"),
                py::arg("wavelength"),
                py::arg("ref_start_range"),
                py::arg("ref_range_pixel_spacing"),
                py::arg("ref_wavelength"))
        .def(py::init<const isce3::product::RadarGridParameters &,
                const isce3::core::LUT2d<double> &, double>(),
                py::arg("rdr_grid"),
                py::arg("doppler"),
                py::arg("wavelength"))
        .def(py::init<const isce3::product::RadarGridParameters &,
                const isce3::product::RadarGridParameters &,
                const isce3::core::LUT2d<double> &, double, double>(),
                py::arg("rdr_grid"),
                py::arg("ref_rdr_grid"),
                py::arg("doppler"),
                py::arg("ref_wavelength"),
                py::arg("wavelength"))
        .def_property("doppler",
                py::overload_cast<>(&ResampSlc::doppler, py::const_),
                &ResampSlc::doppler)
        .def_property("lines_per_tile",
                py::overload_cast<>(&ResampSlc::linesPerTile, py::const_),
                py::overload_cast<size_t>(&ResampSlc::linesPerTile))
        .def_property_readonly("start_range", &ResampSlc::startingRange)
        .def_property_readonly("range_pixel_spacing", &ResampSlc::rangePixelSpacing)
        .def_property_readonly("sensing_start", &ResampSlc::sensingStart)
        .def_property_readonly("prf", &ResampSlc::prf)
        .def_property_readonly("wavelength", &ResampSlc::wavelength)
        .def_property_readonly("ref_start_range", &ResampSlc::refStartingRange)
        .def_property_readonly("ref_range_pixel_spacing", &ResampSlc::refRangePixelSpacing)
        .def_property_readonly("ref_wavelength", &ResampSlc::refWavelength)
        .def("resamp", py::overload_cast<isce3::io::Raster &, isce3::io::Raster &,
                    isce3::io::Raster &, isce3::io::Raster &,
                    int, bool, bool, int , int>(&ResampSlc::resamp),
                py::arg("input_slc"),
                py::arg("output_slc"),
                py::arg("rg_offset_raster"),
                py::arg("az_offset_raster"),
                py::arg("input_band") = 1,
                py::arg("flatten") = false,
                py::arg("is_complex") = true,
                py::arg("row_buffer") = 40,
                py::arg("chip_size") = isce3::core::SINC_ONE)
        .def("resamp", py::overload_cast<const std::string &, const std::string &,
                    const std::string & , const std::string &,
                    int, bool, bool, int, int>(&ResampSlc::resamp),
                py::arg("input_filename"),
                py::arg("output_filename"),
                py::arg("rg_offset_filename"),
                py::arg("az_offset_filename"),
                py::arg("input_band") = 1,
                py::arg("flatten") = false,
                py::arg("is_complex") = true,
                py::arg("row_buffer") = 40,
                py::arg("chip_size") = isce3::core::SINC_ONE)
        ;
}
