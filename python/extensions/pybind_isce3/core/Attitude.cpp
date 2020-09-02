#include "Quaternion.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <isce3/core/DateTime.h>
#include <isce3/io/IH5.h>
#include <isce3/core/Quaternion.h>
#include <isce3/core/Serialization.h>

namespace py = pybind11;
using namespace isce3::core;

void addbinding(pybind11::class_<Attitude>& pyAttitude)
{
    pyAttitude
        .def(py::init<std::vector<double>, std::vector<Quaternion>, DateTime>(),
            py::arg("time"), py::arg("quaternions"), py::arg("epoch"))
        .def("interpolate", &Attitude::interpolate, py::arg("time"))
        .def_static("load_from_h5", [](py::object h5py_group) {
            auto id = h5py_group.attr("id").attr("id").cast<hid_t>();
            isce3::io::IGroup group(id);
            Attitude att;
            loadFromH5(group, att);
            return att;
        },
        "De-serialize Attitude from h5py.Group object",
        py::arg("h5py_group"))
        .def("save_to_h5", [](const Attitude& self, py::object h5py_group) {
                auto id = h5py_group.attr("id").attr("id").cast<hid_t>();
                isce3::io::IGroup group(id);
                isce3::core::saveToH5(group, self);
            },
            "Serialize Attitude to h5py.Group object.",
            py::arg("h5py_group"))

        .def_property_readonly("size", &Attitude::size)
        .def_property_readonly("time", &Attitude::time)
        .def_property_readonly("quaternions", &Attitude::quaternions)
        .def_property_readonly("start_time", &Attitude::startTime)
        .def_property_readonly("end_time", &Attitude::endTime)
        .def_property_readonly("start_datetime", &Attitude::startDateTime)
        .def_property_readonly("end_datetime", &Attitude::endDateTime)
        .def_property_readonly("reference_epoch",
                py::overload_cast<>(&Attitude::referenceEpoch, py::const_))
        ;
}
