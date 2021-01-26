#include "CrossMultiply.h"

#include <complex>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include <isce3/core/EMatrix.h>

using namespace isce3::signal;
namespace py = pybind11;

void addbinding(py::class_<CrossMultiply>& pyCrossMultiply)
{
    using T = isce3::core::EArray2D<std::complex<float>>;

    pyCrossMultiply
            .def(py::init<int, int, int>(), py::arg("nrows"), py::arg("ncols"),
                 py::arg("upsample_factor") = 2,
                 R"(
            Crossmultiplies two aligned SLC images and computes interferogram
            )")
            .def(
                    "crossmultiply",
                    &CrossMultiply::crossmultiply,
                    py::arg("out"), py::arg("ref_slc"), py::arg("sec_slc"), R"(
            Performs cross multiplication on a batch of input signals
            )")

            .def_property_readonly("nrows", &CrossMultiply::nrows)
            .def_property_readonly("ncols", &CrossMultiply::ncols)
            .def_property_readonly("upsample_factor", &CrossMultiply::upsample_factor)
            .def_property_readonly("fftsize", &CrossMultiply::fftsize);
}
