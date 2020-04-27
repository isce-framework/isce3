#include "Orbit.h"

#include <isce/core/Serialization.h>
#include <isce/core/Vector.h>

#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <string>
#include <utility>

namespace py = pybind11;
using isce::core::Orbit;

static py::buffer_info toBuffer(const std::vector<isce::core::Vec3>& buf)
{
    const auto format = py::format_descriptor<double>::format();
    const std::vector<ssize_t> shape  { ssize_t(buf.size()), 3 };
    const std::vector<ssize_t> strides{ sizeof(isce::core::Vec3), sizeof(double) };
    const bool readonly = true;

    return {(void*) buf.data(), sizeof(double), format, 2, shape, strides, readonly};
}

void addbinding(py::class_<Orbit> & pyOrbit)
{
    pyOrbit
        .def_property_readonly("reference_epoch",
                py::overload_cast<>(&Orbit::referenceEpoch, py::const_))
        .def_property_readonly("time", py::overload_cast<>(&Orbit::time, py::const_))
        .def_property_readonly("position", [](const Orbit & self) {
            return py::array{toBuffer(self.position()), py::cast(self)};
        })
        .def_property_readonly("velocity", [](const Orbit & self) {
            return py::array{toBuffer(self.velocity()), py::cast(self)};
        })

        .def_static("load_from_h5", [](py::object h5py_group) {

                auto id = h5py_group.attr("id").attr("id").cast<hid_t>();
                isce::io::IGroup group(id);

                Orbit orbit;
                isce::core::loadFromH5(group, orbit);

                return orbit;
            },
            "De-serialize orbit from h5py.Group object",
            py::arg("h5py_group"))

        // trivial member getters
        .def_property_readonly("spacing",        &Orbit::spacing)
        .def_property_readonly("size",           &Orbit::size)
        .def_property_readonly("start_time",     &Orbit::startTime)
        .def_property_readonly("end_time",       &Orbit::endTime)
        .def_property_readonly("start_datetime", &Orbit::startDateTime)
        .def_property_readonly("mid_datetime",   &Orbit::midDateTime)
        .def_property_readonly("end_datetime",   &Orbit::endDateTime)

        .def("interpolate", [](const Orbit& self, double t) {
                isce::core::Vec3 p, v;
                self.interpolate(&p, &v, t);
                return std::make_pair(p, v);
            },
            "Interpolate platform position and velocity",
            py::arg("t"))
        ;
}
