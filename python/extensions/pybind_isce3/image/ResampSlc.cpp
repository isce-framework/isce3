#include "ResampSlc.h"

#include <pybind11/complex.h>
#include <isce3/core/Constants.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Poly2d.h>
#include <isce3/io/Raster.h>

using isce3::core::Poly2d;
using isce3::image::ResampSlc;

namespace py = pybind11;

void addbinding(py::class_<ResampSlc> & pyResampSlc)
{
    pyResampSlc
        .def(py::init<const isce3::core::LUT2d<double> &, double, double,
                double, double, double, const std::complex<float>>(),
                py::arg("doppler"),
                py::arg("start_range"),
                py::arg("range_pixel_spacing"),
                py::arg("sensing_start"),
                py::arg("prf"),
                py::arg("wavelength"),
                py::arg("invalid_value") = std::complex<float>(0.0, 0.0))
        .def(py::init<const isce3::core::LUT2d<double> &, double, double,
                double, double, double, double, double, double, const std::complex<float>>(),
                py::arg("doppler"),
                py::arg("start_range"),
                py::arg("range_pixel_spacing"),
                py::arg("sensing_start"),
                py::arg("prf"),
                py::arg("wavelength"),
                py::arg("ref_start_range"),
                py::arg("ref_range_pixel_spacing"),
                py::arg("ref_wavelength"),
                py::arg("invalid_value") = std::complex<float>(0.0, 0.0))
        .def(py::init([](const isce3::product::RadarGridParameters & grid,
                const isce3::core::LUT2d<double> & doppler,
                const Poly2d & az_carrier,
                const Poly2d & rg_carrier,
                const std::complex<float> invalid_value,
                const isce3::product::RadarGridParameters * ref_grid) {
                    if (ref_grid) {
                        auto resamp = ResampSlc(grid, *ref_grid, doppler, invalid_value);
                        resamp.azCarrier(az_carrier);
                        resamp.rgCarrier(rg_carrier);
                        return resamp;
                    } else {
                        auto resamp = ResampSlc(grid, doppler, invalid_value);
                        resamp.azCarrier(az_carrier);
                        resamp.rgCarrier(rg_carrier);
                        return resamp;
                    }
                    }),
                    py::arg("rdr_grid"),
                    py::arg("doppler"),
                    py::arg("azimuth_carrier") = Poly2d(),
                    py::arg("range_carrier") =Poly2d(),
                    py::arg("invalid_value") = std::complex<float>(0.0, 0.0),
                    py::arg("ref_rdr_grid") = nullptr)
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
                    int, bool, int , int>(&ResampSlc::resamp),
                py::arg("input_slc"),
                py::arg("output_slc"),
                py::arg("rg_offset_raster"),
                py::arg("az_offset_raster"),
                py::arg("input_band") = 1,
                py::arg("flatten") = false,
                py::arg("row_buffer") = 40,
                py::arg("chip_size") = isce3::core::SINC_ONE,
                R"(
                Resample a SLC

                Parameters
                ----------
                input_slc: isce3.io.Raster
                    Input raster containing SLC to be resampled
                output_slc: isce3.io.Raster
                    Output raster containing resampled SLC
                rg_offset_raster: isce3.io.Raster
                    Raster containing range shift to be applied
                az_offset_raster: isce3.io.Raster
                    Raster containing azimuth shift to be applied
                input_band: int
                    Band of input raster to resample
                flatten: bool
                    Flag to flatten resampled SLC
                row_buffer: int
                    Rows excluded from top/bottom of azimuth raster while searching
                    for min/max row indices of resampled SLC
                )")
        .def("resamp", py::overload_cast<const std::string &, const std::string &,
                    const std::string & , const std::string &,
                    int, bool, int, int>(&ResampSlc::resamp),
                py::arg("input_filename"),
                py::arg("output_filename"),
                py::arg("rg_offset_filename"),
                py::arg("az_offset_filename"),
                py::arg("input_band") = 1,
                py::arg("flatten") = false,
                py::arg("row_buffer") = 40,
                py::arg("chip_size") = isce3::core::SINC_ONE,
                R"(
                Resample a SLC

                Parameters
                ----------
                input_filename: isce3.io.Raster
                    Path of file containing SLC to be resampled
                output_filename: isce3.io.Raster
                    Path of file containing resampled SLC
                rg_offset_filename: isce3.io.Raster
                    Path of file containing range shift to be applied
                az_offset_filename: isce3.io.Raster
                    Path of file containing azimuth shift to be applied
                input_band: int
                    Band of input raster to resample
                flatten: bool
                    Flag to flatten resampled SLC
                row_buffer: int
                    Rows excluded from top/bottom of azimuth raster while searching
                    for min/max row indices of resampled SLC
                )")
        ;
}
