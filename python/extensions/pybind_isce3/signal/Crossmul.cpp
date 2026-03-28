#include "Crossmul.h"

#include <isce3/core/forward.h>
#include <isce3/io/Raster.h>
#include <isce3/product/forward.h>

namespace py = pybind11;

using isce3::io::Raster;
using isce3::signal::Crossmul;

void addbinding(py::class_<Crossmul> & pyCrossmul)
{
    pyCrossmul
        .def(py::init([](const int rg_looks, const int az_looks)
                    {
                        Crossmul crsml;
                        crsml.rangeLooks(rg_looks);
                        crsml.azimuthLooks(az_looks);
                        return crsml;
                    }),
                py::arg("range_looks")=1,
                py::arg("az_looks")=1,
                R"(
    Returns crossmul object with range and and azimuth multilook
    off by default.
                )")
        .def("crossmul", &Crossmul::crossmul,
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("interferogram"),
                py::arg("coherence"),
                py::arg("range_offset") = nullptr,
                py::arg("azimuth_offset") = nullptr, R"(
    Crossmultiply reference and secondary SLCs to generate interferogram and coherence products.

    Parameters
    ----------
    ref_slc: Raster
        Input reference SLC raster
    inputRaster: Raster
        Input secondary SLC raster
    interferogram: Raster
        Output interferogram raster
    coherence: Raster
        Output coherence raster
    range_offset: Raster
        Optional range offset raster usef for flattening and common band filter
    azimuth_offset: Raster
        Optional azimuth offset raster usef for azimuth common band filter
    range_bandwidth: float
        range bandwidth of the reference SLC
    azimuth_bandwidth: float
        azimuth bandwidth of the reference SLC
                )")
        .def("set_dopplers", &Crossmul::doppler,
                py::arg("ref_doppler"),
                py::arg("sec_doppler"))
        .def_property("ref_doppler",
                py::overload_cast<>(&Crossmul::refDoppler, py::const_),
                py::overload_cast<isce3::core::LUT2d<double>>(&Crossmul::refDoppler))
        .def_property("sec_doppler",
                py::overload_cast<>(&Crossmul::secDoppler, py::const_),
                py::overload_cast<isce3::core::LUT2d<double>>(&Crossmul::secDoppler))
        .def_property("ref_sec_offset_starting_range_shift",
                py::overload_cast<>(&Crossmul::startingRangeShift, py::const_),
                py::overload_cast<double>(&Crossmul::startingRangeShift))
        .def_property("range_pixel_spacing",
                py::overload_cast<>(&Crossmul::rangePixelSpacing, py::const_),
                py::overload_cast<double>(&Crossmul::rangePixelSpacing))
        .def_property("window_parameter",
                py::overload_cast<>(&Crossmul::windowParameter, py::const_),
                py::overload_cast<double>(&Crossmul::windowParameter))
        .def_property("sensor_type",
                py::overload_cast<>(&Crossmul::sensorType, py::const_),
                py::overload_cast<std::string>(&Crossmul::sensorType))
        .def_property("window_type",
                py::overload_cast<>(&Crossmul::windowType, py::const_),
                py::overload_cast<std::string>(&Crossmul::windowType))
        .def_property("do_common_range_band_filter",
                py::overload_cast<>(&Crossmul::doCommonRangeBandFilter, py::const_),
                py::overload_cast<bool>(&Crossmul::doCommonRangeBandFilter))
        .def_property("do_common_azimuth_band_filter",
                py::overload_cast<>(&Crossmul::doCommonAzimuthBandFilter, py::const_),
                py::overload_cast<bool>(&Crossmul::doCommonAzimuthBandFilter))
        .def_property("do_flatten",
                py::overload_cast<>(&Crossmul::doFlatten, py::const_),
                py::overload_cast<bool>(&Crossmul::doFlatten))
        .def_property("range_bandwidth",
                py::overload_cast<>(&Crossmul::rangeBandwidth, py::const_),
                py::overload_cast<double>(&Crossmul::rangeBandwidth))
        .def_property("ref_start_range",
                py::overload_cast<>(&Crossmul::refStartRange, py::const_),
                py::overload_cast<double>(&Crossmul::refStartRange))
        .def_property("ref_start_azimuth_time",
                py::overload_cast<>(&Crossmul::refStartAzimuthTime, py::const_),
                py::overload_cast<double>(&Crossmul::refStartAzimuthTime))
        .def_property("sec_start_range",
                py::overload_cast<>(&Crossmul::secStartRange, py::const_),
                py::overload_cast<double>(&Crossmul::secStartRange))
        .def_property("sec_start_azimuth_time",
                py::overload_cast<>(&Crossmul::secStartAzimuthTime, py::const_),
                py::overload_cast<double>(&Crossmul::secStartAzimuthTime))
        .def_property("azimuth_bandwidth",
                py::overload_cast<>(&Crossmul::azimuthBandwidth, py::const_),
                py::overload_cast<double>(&Crossmul::azimuthBandwidth))
        .def_property("prf",
                py::overload_cast<>(&Crossmul::prf, py::const_),
                py::overload_cast<double>(&Crossmul::prf))
        .def_property("wavelength",
                py::overload_cast<>(&Crossmul::wavelength, py::const_),
                py::overload_cast<double>(&Crossmul::wavelength))
        .def_property("range_looks",
                py::overload_cast<>(&Crossmul::rangeLooks, py::const_),
                py::overload_cast<int>(&Crossmul::rangeLooks))
        .def_property("az_looks",
                py::overload_cast<>(&Crossmul::azimuthLooks, py::const_),
                py::overload_cast<int>(&Crossmul::azimuthLooks))
        .def_property("oversample_factor",
                py::overload_cast<>(&Crossmul::oversampleFactor, py::const_),
                py::overload_cast<size_t>(&Crossmul::oversampleFactor))
        .def_property("lines_per_block",
                py::overload_cast<>(&Crossmul::linesPerBlock, py::const_),
                py::overload_cast<size_t>(&Crossmul::linesPerBlock))
        .def_property_readonly("multilook_enabled", &Crossmul::multiLookEnabled)
        .def_property_readonly("processed_range_bandwidth", &Crossmul::processedRangeBandwidth)
        .def_property_readonly("processed_azimuth_bandwidth", &Crossmul::processedAzimuthBandwidth)
        ;
}
