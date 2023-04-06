#include "LUT1d.h"

#include <valarray>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

#include <isce3/core/LUT2d.h>

namespace py = pybind11;

using isce3::core::LUT1d;

template<typename T>
void addbinding(py::class_<LUT1d<T>> &pyLUT1d)
{
    pyLUT1d
        .def(py::init<>())
        .def(py::init<const std::valarray<double>&, const std::valarray<T>&, bool>(),
                py::arg("coords"),
                py::arg("values"),
                py::arg("extraploate")=true)
        .def_property_readonly("size", &LUT1d<T>::size)
        .def("coordinates", (std::valarray<double>& (LUT1d<T>::*)()) &LUT1d<T>::coords,
                "coords getter")
        .def("values", (std::valarray<T>& (LUT1d<T>::*)()) &LUT1d<T>::values,
                "values getter")
        .def("eval", py::overload_cast<double>(&LUT1d<T>::eval, py::const_))
        .def("eval", py::overload_cast<const Eigen::Ref<const Eigen::ArrayXd>& >(
	        &LUT1d<T>::eval, py::const_))
        ;
}

template void addbinding(py::class_<LUT1d<double>> &pyLUT1d);

// end of file
