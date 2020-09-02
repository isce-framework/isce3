#include "Basis.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/EulerAngles.h>
#include <isce3/core/Quaternion.h>
#include <isce3/core/Vector.h>

namespace py = pybind11;
using namespace isce3::core;

void addbinding(pybind11::class_<Basis> &pyBasis)
{
    pyBasis
        .def(py::init<const Vec3&, const Vec3&, const Vec3&>(),
            py::arg("x0"), py::arg("x1"), py::arg("x2"),
            "Basis from three unit vectors")
        .def(py::init<const Vec3&, const Vec3&>(),
            py::arg("position"), py::arg("velocity"),
            "Geocentric TCN basis from position and velocity")
        .def("project", &Basis::project)
        .def("combine", [](const Basis & self, const Vec3 & x) {
            Vec3 out;
            self.combine(x, out);
            return out;
        })
        .def("asarray", [](const Basis& self) { return self.toRotationMatrix(); },
            "Returns basis vectors as columns of a 2D array.")
        .def_property_readonly("x0", py::overload_cast<>(&Basis::x0, py::const_))
        .def_property_readonly("x1", py::overload_cast<>(&Basis::x1, py::const_))
        .def_property_readonly("x2", py::overload_cast<>(&Basis::x2, py::const_))
    ;
}

void addbinding_basis(pybind11::module& mod)
{
    mod
        .def("velocity_eci", &velocityECI, py::arg("position"),
            py::arg("velocity_ecf"))
        .def("geodetic_tcn", &geodeticTCN, py::arg("position"),
            py::arg("velocity"), py::arg("ellipsoid"))
        // geodetic
        .def("factored_ypr",
            py::overload_cast<const Quaternion&, const Vec3&,
                const Vec3&, const Ellipsoid&>(&factoredYawPitchRoll),
            py::arg("quaternion"), py::arg("position"), py::arg("velocity"),
            py::arg("ellipsoid"))
        // geocentric
        .def("factored_ypr",
            py::overload_cast<const Quaternion&, const Vec3&, const Vec3&>(
                &factoredYawPitchRoll),
            py::arg("quaternion"), py::arg("position"), py::arg("velocity"))
    ;
}
