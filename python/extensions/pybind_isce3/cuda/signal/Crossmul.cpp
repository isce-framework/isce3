#include "Crossmul.h"

#include <isce3/core/forward.h>
#include <isce3/io/Raster.h>
#include <isce3/product/forward.h>

namespace py = pybind11;

using isce3::io::Raster;
using isce3::cuda::signal::gpuCrossmul;

void addbinding(py::class_<gpuCrossmul> & pyCrossmul)
{
    pyCrossmul
        .def(py::init([](const int rg_looks, const int az_looks)
                    {
                        gpuCrossmul crsml;
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
        .def("crossmul", &gpuCrossmul::crossmul,
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("interferogram"),
                py::arg("coherence"),
                py::arg("range_offset") = nullptr, R"(
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
    interferogram: Raster
        Optional range offset raster usef for flattening
                )")
        .def("set_dopplers", &gpuCrossmul::doppler,
                py::arg("ref_doppler"),
                py::arg("sec_doppler"))
        .def_property("ref_doppler",
                py::overload_cast<>(&gpuCrossmul::refDoppler, py::const_),
                py::overload_cast<isce3::core::LUT1d<double>>(&gpuCrossmul::refDoppler))
        .def_property("sec_doppler",
                py::overload_cast<>(&gpuCrossmul::secDoppler, py::const_),
                py::overload_cast<isce3::core::LUT1d<double>>(&gpuCrossmul::secDoppler))
        .def_property("ref_sec_offset_starting_range_shift",
                py::overload_cast<>(&gpuCrossmul::startingRangeShift, py::const_),
                py::overload_cast<double>(&gpuCrossmul::startingRangeShift))
        .def_property("range_pixel_spacing",
                py::overload_cast<>(&gpuCrossmul::rangePixelSpacing, py::const_),
                py::overload_cast<double>(&gpuCrossmul::rangePixelSpacing))
        .def_property("wavelength",
                py::overload_cast<>(&gpuCrossmul::wavelength, py::const_),
                py::overload_cast<double>(&gpuCrossmul::wavelength))
        .def_property("range_looks",
                py::overload_cast<>(&gpuCrossmul::rangeLooks, py::const_),
                py::overload_cast<int>(&gpuCrossmul::rangeLooks))
        .def_property("az_looks",
                py::overload_cast<>(&gpuCrossmul::azimuthLooks, py::const_),
                py::overload_cast<int>(&gpuCrossmul::azimuthLooks))
        .def_property("oversample_factor",
                py::overload_cast<>(&gpuCrossmul::oversampleFactor, py::const_),
                py::overload_cast<size_t>(&gpuCrossmul::oversampleFactor))
        .def_property("lines_per_block",
                py::overload_cast<>(&gpuCrossmul::linesPerBlock, py::const_),
                py::overload_cast<size_t>(&gpuCrossmul::linesPerBlock))
        .def_property_readonly("multilook_enabled", &gpuCrossmul::multiLookEnabled)
        ;
}
