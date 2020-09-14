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
        .def(py::init<>())
        .def("crossmul", py::overload_cast<Raster&, Raster&, Raster&, Raster&, Raster&>(&Crossmul::crossmul),
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("rg_offset"),
                py::arg("coherence"),
                py::arg("interferogram"))
        .def("crossmul", py::overload_cast<Raster&, Raster&, Raster&, Raster&>(&Crossmul::crossmul),
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("coherence"),
                py::arg("interferogram"))
        .def("crossmul", py::overload_cast<Raster&, Raster&, Raster&>(&Crossmul::crossmul),
                py::arg("ref_slc"),
                py::arg("sec_slc"),
                py::arg("interferogram"))
        .def("set_dopplers", &Crossmul::doppler,
                py::arg("ref_doppler"),
                py::arg("sec_doppler"))
        .def_property("prf", nullptr, &Crossmul::prf)
        .def_property("rg_sampling_freq", nullptr, &Crossmul::rangeSamplingFrequency)
        .def_property("rg_bw", nullptr, &Crossmul::rangeBandwidth)
        .def_property("rg_pixel_spacing", nullptr, &Crossmul::rangePixelSpacing)
        .def_property("wavelength", nullptr, &Crossmul::wavelength)
        .def_property("common_az_bw", nullptr, &Crossmul::commonAzimuthBandwidth)
        .def_property("beta", nullptr, &Crossmul::beta)
        .def_property("rg_looks", nullptr, &Crossmul::rangeLooks)
        .def_property("az_looks", nullptr, &Crossmul::azimuthLooks)
        .def_property("filter_az", nullptr, &Crossmul::doCommonAzimuthbandFiltering)
        .def_property("filter_rg", nullptr, &Crossmul::doCommonRangebandFiltering)
        ;
}
