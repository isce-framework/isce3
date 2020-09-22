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
                        crsml.doCommonRangeBandFilter(false);
                        crsml.doCommonAzimuthBandFilter(false);
                        return crsml;
                    }),
                py::arg("range_looks")=1,
                py::arg("az_looks")=1,
                R"(
    Returns crossmul object with range and and azimuth multilook
    off by default.
                )")
        .def("set_rg_filter", [](gpuCrossmul & self,
                    const double rg_sampling_freq,
                    const double rg_bw,
                    const double rg_pixel_spacing,
                    const double wavelength)
                {
                    self.doCommonRangeBandFilter(true);
                    self.rangeSamplingFrequency(rg_sampling_freq);
                    self.rangeBandwidth(rg_bw);
                    self.rangePixelSpacing(rg_pixel_spacing);
                    self.wavelength(wavelength);
                },
                py::arg("range_sampling_freq"),
                py::arg("range_bandwidth"),
                py::arg("range_pixel_spacing"),
                py::arg("wavelength"),
                R"(
    Enables range band filtering and sets filtering parameters.
                )")
        .def("set_az_filter", [](gpuCrossmul & self,
                    const double prf,
                    const double common_az_bw,
                    const double beta)
                {
                    self.doCommonAzimuthBandFilter(true);
                    self.prf(prf);
                    self.commonAzimuthBandwidth(common_az_bw);
                    self.beta(beta);
                },
                py::arg("prf"),
                py::arg("common_az_bandwidth"),
                py::arg("beta"),
                R"(
    Enables azimuth band filtering and sets filtering parameters.
                )")
        .def("crossmul", py::overload_cast<Raster&, Raster&, Raster&, Raster&, Raster&>(&gpuCrossmul::crossmul),
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("range_offset"),
                py::arg("coherence"),
                py::arg("interferogram"))
        .def("crossmul", py::overload_cast<Raster&, Raster&, Raster&, Raster&>(&gpuCrossmul::crossmul),
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("coherence"),
                py::arg("interferogram"))
        .def("set_dopplers", &gpuCrossmul::doppler,
                py::arg("ref_doppler"),
                py::arg("sec_doppler"))
        .def_property("prf",
                py::overload_cast<>(&gpuCrossmul::prf, py::const_),
                py::overload_cast<double>(&gpuCrossmul::prf))
        .def_property("range_sampling_freq",
                py::overload_cast<>(&gpuCrossmul::rangeSamplingFrequency, py::const_),
                py::overload_cast<double>(&gpuCrossmul::rangeSamplingFrequency))
        .def_property("range_bandwidth",
                py::overload_cast<>(&gpuCrossmul::rangeBandwidth, py::const_),
                py::overload_cast<double>(&gpuCrossmul::rangeBandwidth))
        .def_property("range_pixel_spacing",
                py::overload_cast<>(&gpuCrossmul::rangePixelSpacing, py::const_),
                py::overload_cast<double>(&gpuCrossmul::rangePixelSpacing))
        .def_property("wavelength",
                py::overload_cast<>(&gpuCrossmul::wavelength, py::const_),
                py::overload_cast<double>(&gpuCrossmul::wavelength))
        .def_property("common_az_bandwidth",
                py::overload_cast<>(&gpuCrossmul::commonAzimuthBandwidth, py::const_),
                py::overload_cast<double>(&gpuCrossmul::commonAzimuthBandwidth))
        .def_property("beta",
                py::overload_cast<>(&gpuCrossmul::beta, py::const_),
                py::overload_cast<double>(&gpuCrossmul::beta))
        .def_property("range_looks",
                py::overload_cast<>(&gpuCrossmul::rangeLooks, py::const_),
                py::overload_cast<int>(&gpuCrossmul::rangeLooks))
        .def_property("az_looks",
                py::overload_cast<>(&gpuCrossmul::azimuthLooks, py::const_),
                py::overload_cast<int>(&gpuCrossmul::azimuthLooks))
        .def_property("filter_az",
                py::overload_cast<>(&gpuCrossmul::doCommonAzimuthBandFilter, py::const_),
                py::overload_cast<bool>(&gpuCrossmul::doCommonAzimuthBandFilter))
        .def_property("filter_rg",
                py::overload_cast<>(&gpuCrossmul::doCommonRangeBandFilter, py::const_),
                py::overload_cast<bool>(&gpuCrossmul::doCommonRangeBandFilter))
        ;
}
