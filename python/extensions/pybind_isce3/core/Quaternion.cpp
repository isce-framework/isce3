#include "Quaternion.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include <isce/core/DenseMatrix.h>
#include <isce/core/Quaternion.h>

using isce::core::Quaternion;

namespace py = pybind11;

void addbinding(pybind11::class_<Quaternion>& pyQuaternion)
{
    pyQuaternion
        .def(py::init([](const py::array_t<double>& t,
                         const py::array_t<double, py::array::c_style>& q) {
            if (t.ndim() != 1)
                throw std::length_error("Time array must be 1D.");
            if (q.ndim() != 2)
                throw std::length_error("Quaternion array must be 2D.");
            if (t.shape(0) != q.shape(0))
                throw std::length_error("Shapes must match.");
            if (q.shape(1) != 4)
                throw std::length_error("Quaternions must have 4 elements.");
            auto n = t.shape(0);
            std::vector<double> tv(t.data(), t.data() + n);
            std::vector<double> qv(q.data(), q.data() + 4 * n);
            return Quaternion(tv, qv);
        }),
        py::arg("time"), py::arg("quaternions"))

        .def("rotmat", [](Quaternion & self, double t) {
            return self.rotmat(t, "");
        })
        ;
}
