#include "Quaternion.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <isce3/core/DenseMatrix.h>
#include <isce3/core/EulerAngles.h>
#include <isce3/io/IH5.h>
#include <isce3/core/Quaternion.h>

namespace py = pybind11;
using namespace isce3::core;

void addbinding(pybind11::class_<Quaternion>& pyQuaternion)
{
    pyQuaternion
        .def(py::init<double,double,double,double>(),
            py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
        .def(py::init<const EulerAngles&>(), py::arg("ypr"))
        .def(py::init<Mat3>(), py::arg("rotation_matrix"))
        .def("to_rotation_matrix", &Quaternion::toRotationMatrix)
        .def_property_readonly("w", py::overload_cast<>(&Quaternion::w, py::const_))
        .def_property_readonly("x", py::overload_cast<>(&Quaternion::x, py::const_))
        .def_property_readonly("y", py::overload_cast<>(&Quaternion::y, py::const_))
        .def_property_readonly("z", py::overload_cast<>(&Quaternion::z, py::const_))
        ;
}
