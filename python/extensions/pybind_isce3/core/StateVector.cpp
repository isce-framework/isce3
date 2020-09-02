#include "StateVector.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/DenseMatrix.h>

namespace py = pybind11;

using namespace isce3::core;

void addbinding(py::class_<StateVector> &pyStateVector)
{
    pyStateVector
        .def(py::init<const DateTime&, const Vec3&, const Vec3&>(),
            py::arg("datetime"), py::arg("position"), py::arg("velocity"))
        .def_readonly("datetime", &StateVector::datetime)
        .def_readonly("position", &StateVector::position)
        .def_readonly("velocity", &StateVector::velocity)
    ;
}
