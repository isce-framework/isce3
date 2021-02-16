#include "EulerAngles.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
// XXX matrix/vector types only forward declared in isce3/core/EulerAngles.h
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Quaternion.h>
#include <isce3/core/Vector.h>

using namespace isce3::core;

namespace py = pybind11;

void addbinding(pybind11::class_<EulerAngles>& pyEulerAngles)
{
    pyEulerAngles
            // Constructors
            .def(py::init<double, double, double>(), py::arg("yaw"),
                    py::arg("pitch"), py::arg("roll"))
            .def(py::init<Quaternion>(), py::arg("quaternion"))
            .def(py::init<Mat3>(), py::arg("rotation_matrix"))

            // Regular Methods
            .def("to_rotation_matrix", &EulerAngles::toRotationMatrix)
            .def("to_quaternion", &EulerAngles::toQuaternion,
                    "Convert to isce3 Quaternion object")
            .def("is_approx", &EulerAngles::isApprox,
                    "True if self and 'other' is close within 'prec'",
                    py::arg("other"), py::arg("prec") = 1e-7)
            .def("rotate", &EulerAngles::rotate,
                    "Rotate a 3-D vector by self in YPR order",
                    py::arg("vector3d"))

            // Dunder/Magic Methods
            .def("__repr__",
                    [](const EulerAngles& ea) {
                        std::stringstream os;
                        os << "<EulerAngles(" << ea.yaw() << ',' << ea.pitch()
                           << ',' << ea.roll() << ")>";
                        return os.str();
                    })

            .def("__call__",
                    [](const EulerAngles& ea) {
                        Eigen::Vector3d vec(ea.yaw(), ea.pitch(), ea.roll());
                        return vec;
                    })

            // Operators
            .def(py::self += py::self)
            .def(py::self -= py::self)
            .def(py::self *= py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)

            // Properties
            .def_property_readonly(
                    "yaw", py::overload_cast<>(&EulerAngles::yaw, py::const_))
            .def_property_readonly("pitch",
                    py::overload_cast<>(&EulerAngles::pitch, py::const_))
            .def_property_readonly("roll",
                    py::overload_cast<>(&EulerAngles::roll, py::const_));
}
