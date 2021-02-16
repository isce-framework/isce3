#include "Quaternion.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <sstream>

#include <isce3/core/DenseMatrix.h>
#include <isce3/core/EulerAngles.h>
#include <isce3/core/Quaternion.h>
#include <isce3/io/IH5.h>

namespace py = pybind11;
using namespace isce3::core;

void addbinding(pybind11::class_<Quaternion>& pyQuaternion)
{
    pyQuaternion
            // Constructors
            .def(py::init<>())
            .def(py::init<const Eigen::Vector4d&>(),
                    py::arg("quaternion_vector"))
            .def(py::init<const Eigen::Vector3d&>(), py::arg("vector3d"))
            .def(py::init<double, const Eigen::Vector3d&>(), py::arg("angle"),
                    py::arg("axis"))
            .def(py::init<double, double, double>(), py::arg("yaw"),
                    py::arg("pitch"), py::arg("roll"))
            .def(py::init<double, double, double, double>(), py::arg("w"),
                    py::arg("x"), py::arg("y"), py::arg("z"))
            .def(py::init<const EulerAngles&>(), py::arg("ypr"))
            .def(py::init<Mat3>(), py::arg("rotation_matrix"))
            .def(py::init<const Quaternion&>(), py::arg("quaternion"))

            // Dunder/Magic Methods
            .def("__repr__",
                    [](const Quaternion& q) {
                        std::stringstream os;
                        os << "<Quaternion(" << q.w() << ',' << q.x() << ','
                           << q.y() << ',' << q.z() << ")>";
                        return os.str();
                    })

            .def("__call__",
                    [](const Quaternion& q) {
                        Eigen::Vector4d vec(q.w(), q.x(), q.y(), q.z());
                        return vec;
                    })

            // Regular Methods
            .def("to_rotation_matrix", &Quaternion::toRotationMatrix)
            .def("rotate",
                    py::overload_cast<const Eigen::Vector3d&>(
                            &Quaternion::rotate, py::const_),
                    "Rotate a 3-D vector by quaternions", py::arg("vector3d"))
            .def("to_ypr", py::overload_cast<>(&Quaternion::toYPR, py::const_),
                    "Convert quaternions to an array of (yaw, pitch, roll) "
                    "angles in radians")
            .def("to_euler_angles",
                    py::overload_cast<>(&Quaternion::toEulerAngles, py::const_),
                    "Convert quaternion to ISCE3 EulerAngles object")
            .def("conjugate",
                    [](const Quaternion& q) {
                        return static_cast<Quaternion>(q.conjugate());
                    })
            .def(
                    "slerp",
                    [](const Quaternion& q, double t, const Quaternion& other) {
                        return static_cast<Quaternion>(q.slerp(t, other));
                    },
                    "Spherical linear interpolation between two quaternions "
                    "'self' and 'quatern' at 't' in [0,1]",
                    py::arg("t"), py::arg("quatern"))
            .def(
                    "dot",
                    [](const Quaternion& q, const Quaternion& other) {
                        return q.dot(other);
                    },
                    "Dot product of self and other quaternion 'quatern'",
                    py::arg("quatern"))
            .def(
                    "is_approx",
                    [](const Quaternion& q, const Quaternion& other,
                            double prec = 1e-12) {
                        return q.isApprox(other, prec);
                    },
                    "true if self is equal to 'quatern' within the 'prec'",
                    py::arg("quatern"), py::arg("prec") = 1e-12)

            // Operators
            .def(
                    "__mul__",
                    [](const Quaternion& q, const Quaternion& other) {
                        return static_cast<Quaternion>(q * other);
                    },
                    py::is_operator())

            // Properties
            .def_property_readonly(
                    "w", py::overload_cast<>(&Quaternion::w, py::const_))
            .def_property_readonly(
                    "x", py::overload_cast<>(&Quaternion::x, py::const_))
            .def_property_readonly(
                    "y", py::overload_cast<>(&Quaternion::y, py::const_))
            .def_property_readonly(
                    "z", py::overload_cast<>(&Quaternion::z, py::const_))
            .def_property_readonly(
                    "vec", py::overload_cast<>(&Quaternion::vec, py::const_));
}
