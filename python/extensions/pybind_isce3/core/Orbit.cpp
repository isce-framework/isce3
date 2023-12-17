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

void addbinding(pybind11::enum_<isce3::core::OrbitInterpMethod> & pyOrbitInterpMethod)
{
    pyOrbitInterpMethod
        .value("HERMITE", isce3::core::OrbitInterpMethod::Hermite)
        .value("LEGENDRE", isce3::core::OrbitInterpMethod::Legendre);
}


void addbinding(py::class_<Orbit> & pyOrbit)
{
    pyOrbit
        .def(py::init<std::vector<StateVector>>())
        .def(py::init<std::vector<StateVector>, isce3::core::DateTime>())
        .def_property_readonly("reference_epoch",
                py::overload_cast<>(&Orbit::referenceEpoch, py::const_))
        // add a function rather than just a settable property since
        // self.time is also modified
        .def("update_reference_epoch",
            py::overload_cast<const DateTime&>(&Orbit::referenceEpoch),
            R"(
            Set reference epoch and modify time tags so that
                self.reference_epoch + TimeDelta(self.time[i])
            remains invariant for all i in range(len(self.time)).)",
            py::arg("reference_epoch"))
        .def("copy", [](const Orbit& self) { return Orbit(self); })
        .def("__copy__", [](const Orbit& self) { return Orbit(self); })
        .def("__deepcopy__", [](const Orbit& self,  py::dict) {
                return Orbit(self);
            }, py::arg("memo"))
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
        .def_property_readonly("mid_time",       &Orbit::midTime)
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

        .def("contains", &Orbit::contains,
            "Check if time falls in the valid interpolation domain.",
            py::arg("time"))

        .def("crop", &Orbit::crop,
            R"(
            Create a new Orbit containing data in the requested interval

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
            isce3.core.Orbit
                Orbit object with data containing start & end times)",
            py::arg("start"), py::arg("end"), py::arg("npad") = 0)
        .def("set_interp_method", [](Orbit& self, const std::string& method) {
            if (method == "Hermite") {
                self.interpMethod(OrbitInterpMethod::Hermite);
            } else if (method == "Legendre") {
                self.interpMethod(OrbitInterpMethod::Legendre);
            } else {
                throw std::invalid_argument("unexpected orbit interpolation method '" + method + "'");
            }
        }, R"(
        Set interpolation method.

        Parameters
        ----------
        method : {'Hermite', 'Legendre'}
            The method for interpolating orbit state vectors (cubic
            Hermite spline interpolation or eighth-order Legendre
            polynomial interpolation).
        )",
        py::arg("method"))
        .def("get_interp_method", [](Orbit& self) {
            return self.interpMethod();
        }, R"(
        Get interpolation method.

        Returns
        -------
        orbit_method: isce3.core.OrbitInterpMethod
            The method for interpolating orbit state vectors (cubic
            Hermite spline interpolation or eighth-order Legendre
            polynomial interpolation).
        )");
}