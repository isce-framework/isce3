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

        // add a function rather than just a settable property since
        // self.time is also modified
        .def("update_reference_epoch",
            py::overload_cast<const DateTime&>(&Attitude::referenceEpoch),
            R"(
            Set reference epoch and modify time tags so that
                self.reference_epoch + TimeDelta(self.time[i])
            remains invariant for all i in range(len(self.time)).)",
            py::arg("reference_epoch"))

        .def("copy", [](const Attitude& self) { return Attitude(self);})
        .def("__copy__", [](const Attitude& self) { return Attitude(self);})
        .def("__deepcopy__", [](const Attitude& self, py::dict) {
                return Attitude(self);
            }, py::arg("memo"))

        .def_property_readonly("size", &Attitude::size)
        .def_property_readonly("time", &Attitude::time)
        .def_property_readonly("quaternions", &Attitude::quaternions)
        .def_property_readonly("start_time", &Attitude::startTime)
        .def_property_readonly("end_time", &Attitude::endTime)
        .def_property_readonly("start_datetime", &Attitude::startDateTime)
        .def_property_readonly("end_datetime", &Attitude::endDateTime)
        .def_property_readonly("reference_epoch",
                py::overload_cast<>(&Attitude::referenceEpoch, py::const_))
        .def("contains", &Attitude::contains,
            "Check if time falls in the valid interpolation domain.",
            py::arg("time"))

        .def("crop", &Attitude::crop,
            R"(
            Create a new Attitude containing data in the requested interval

            Parameters
            ----------
            start : isce3.core.DateTime
                Beginning of time interval
            end : isce3.core.DateTime
                End of time interval
            npad : int, optional
                Minimal number of state vectors to include past each of
                the given time bounds (useful to guarantee adequate
                support for interpolation).
            Returns
            -------
            isce3.core.Attitude
                Attitude object with data containing start & end times)",
            py::arg("start"), py::arg("end"), py::arg("npad") = 0)
        ;
}
