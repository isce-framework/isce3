#include "flatten.h"

#include <complex>
#include <limits>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <isce3/core/EMatrix.h>

namespace py = pybind11;

void addbinding_flatten(pybind11::module& m)
{
    m.def("flatten", &isce3::signal::flatten, py::arg("ifgram"),
          py::arg("range_offset"), py::arg("range_spacing"),
          py::arg("wavelength"),
          R"(This function flattens the input interferogram by removing the 
                phase component due to slant range difference between the 
                interferometric (i.e, slant range offset) between the 
                interferometric pair.
             )");
}
