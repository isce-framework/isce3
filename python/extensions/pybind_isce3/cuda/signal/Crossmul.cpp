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
        .def(py::init<>())
        .def("crossmul", py::overload_cast<Raster&, Raster&, Raster&, Raster&, Raster&>(&gpuCrossmul::crossmul),
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("rg_offset"),
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
        .def_property("prf", nullptr, &gpuCrossmul::prf)
        .def_property("rg_sampling_freq", nullptr, &gpuCrossmul::rangeSamplingFrequency)
        .def_property("rg_bw", nullptr, &gpuCrossmul::rangeBandwidth)
        .def_property("rg_pixel_spacing", nullptr, &gpuCrossmul::rangePixelSpacing)
        .def_property("wavelength", nullptr, &gpuCrossmul::wavelength)
        .def_property("common_az_bw", nullptr, &gpuCrossmul::commonAzimuthBandwidth)
        .def_property("beta", nullptr, &gpuCrossmul::beta)
        .def_property("rg_looks", nullptr, &gpuCrossmul::rangeLooks)
        .def_property("az_looks", nullptr, &gpuCrossmul::azimuthLooks)
        .def_property("filter_az", nullptr, &gpuCrossmul::doCommonAzimuthBandFiltering)
        .def_property("filter_rg", nullptr, &gpuCrossmul::doCommonRangeBandFiltering)
        ;
}
