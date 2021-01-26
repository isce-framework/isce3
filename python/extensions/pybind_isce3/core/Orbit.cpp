#include "Orbit.h"

#include <isce3/core/Serialization.h>
#include <isce3/core/Vector.h>

#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <pybind_isce3/roarray.h>

#include <string>
#include <utility>

namespace py = pybind11;
using isce3::core::Orbit;
using isce3::core::StateVector;

static py::buffer_info toBuffer(const std::vector<isce3::core::Vec3>& buf)
{
    const auto format = py::format_descriptor<double>::format();
    const std::vector<ssize_t> shape  { ssize_t(buf.size()), 3 };
    const std::vector<ssize_t> strides{ sizeof(isce3::core::Vec3), sizeof(double) };
    const bool readonly = true;

    return {(void*) buf.data(), sizeof(double), format, 2, shape, strides, readonly};
}

void addbinding(py::class_<Orbit> & pyOrbit)
{
    pyOrbit
        .def(py::init<std::vector<StateVector>>())
        .def(py::init<std::vector<StateVector>, isce3::core::DateTime>())
        .def_property_readonly("reference_epoch",
                py::overload_cast<>(&Orbit::referenceEpoch, py::const_))
        .def_property_readonly("time", py::overload_cast<>(&Orbit::time, py::const_))
        .def_property_readonly("position", [](const Orbit & self) {
            return py::roarray(toBuffer(self.position()), py::cast(self));
        })
        .def_property_readonly("velocity", [](const Orbit & self) {
            return py::roarray(toBuffer(self.velocity()), py::cast(self));
        })

        .def_static("load_from_h5", [](py::object h5py_group) {

                auto id = h5py_group.attr("id").attr("id").cast<hid_t>();
                isce3::io::IGroup group(id);

                Orbit orbit;
                isce3::core::loadFromH5(group, orbit);

                return orbit;
            },
            "De-serialize orbit from h5py.Group object",
            py::arg("h5py_group"))

        .def("save_to_h5", [](const Orbit& self, py::object h5py_group) {
                auto id = h5py_group.attr("id").attr("id").cast<hid_t>();
                isce3::io::IGroup group(id);
                isce3::core::saveToH5(group, self);
            },
            "Serialize Orbit to h5py.Group object.",
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
                isce3::core::Vec3 p, v;
                self.interpolate(&p, &v, t);
                return std::make_pair(p, v);
            },
            "Interpolate platform position and velocity",
            py::arg("t"))
        ;
}
