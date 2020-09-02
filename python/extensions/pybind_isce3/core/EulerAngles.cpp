#include "EulerAngles.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
// XXX matrix/vector types only forward declared in isce3/core/EulerAngles.h
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Vector.h>
#include <isce3/core/Quaternion.h>

using namespace isce3::core;

namespace py = pybind11;

void addbinding(pybind11::class_<EulerAngles>& pyEulerAngles)
{
    pyEulerAngles
        .def(py::init<double, double, double>(), py::arg("yaw"),
            py::arg("pitch"), py::arg("roll"))
        .def(py::init<Quaternion>(), py::arg("quaternion"))
        .def(py::init<Mat3>(), py::arg("rotation_matrix"))
        .def("to_rotation_matrix", &EulerAngles::toRotationMatrix)
        .def_property_readonly("yaw",
            py::overload_cast<>(&EulerAngles::yaw, py::const_))
        .def_property_readonly("pitch",
            py::overload_cast<>(&EulerAngles::pitch, py::const_))
        .def_property_readonly("roll",
            py::overload_cast<>(&EulerAngles::roll, py::const_))
    ;
}
