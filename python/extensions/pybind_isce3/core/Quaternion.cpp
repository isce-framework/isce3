#include "Quaternion.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include <isce3/core/DenseMatrix.h>
#include <isce3/io/IH5.h>
#include <isce3/core/Quaternion.h>
#include <isce3/core/Serialization.h>

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
        .def("ypr", [](Quaternion & self, double t) {
            double yaw{0}, pitch{0}, roll{0};
            self.ypr(t, yaw, pitch, roll);
            auto out = py::tuple(3);
            out[0] = yaw;
            out[1] = pitch;
            out[2] = roll;
            return out;
        })
        .def_static("load_from_h5", [](py::object h5py_group) {
            auto id = h5py_group.attr("id").attr("id").cast<hid_t>();
            isce::io::IGroup group(id);
            Quaternion q;
            isce::core::loadFromH5(group, q);
            return q;
        },
        "De-serialize Quaternion from h5py.Group object",
        py::arg("h5py_group"))

        .def_property_readonly("time", &Quaternion::time)
        .def_property_readonly("qvec", [](const Quaternion& self) {
            using namespace Eigen;
            using T = Array<double, Dynamic, Dynamic, RowMajor>;
            return Map<const T>(self.qvec().data(), self.nVectors(), 4);
        })
        ;
}
