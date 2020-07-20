#include "TimeDelta.h"

#include <memory>
#include <pybind11/chrono.h>
#include <pybind11/operators.h>
#include <string>

namespace py = pybind11;

using isce3::core::TimeDelta;

void addbinding(py::class_<TimeDelta> & pyTimeDelta)
{
    pyTimeDelta
        .def(py::init<>())
        .def(py::init<double>(), py::arg("seconds"))
        .def(py::init<int, int, int>(),
                py::arg("hours"),
                py::arg("minutes"),
                py::arg("seconds"))
        .def(py::init<int, int, double>(),
                py::arg("hours"),
                py::arg("minutes"),
                py::arg("seconds"))
        .def(py::init<int, int, int, double>(),
                py::arg("hours"),
                py::arg("minutes"),
                py::arg("seconds"),
                py::arg("frac"))
        .def(py::init<int, int, int, int, double>(),
                py::arg("days"),
                py::arg("hours"),
                py::arg("minutes"),
                py::arg("seconds"),
                py::arg("frac"))
        .def(py::init([](const std::chrono::duration<int, std::micro> & duration)
                {
                    return TimeDelta(1e-6 * duration.count());
                }))
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * double())
        .def(double() * py::self)
        .def(py::self *= double())
        .def(py::self / double())
        .def(py::self /= double())
        .def_readwrite("days", &TimeDelta::days)
        .def_readwrite("hours", &TimeDelta::hours)
        .def_readwrite("minutes", &TimeDelta::minutes)
        .def_readwrite("seconds", &TimeDelta::seconds)
        .def_readwrite("frac", &TimeDelta::frac)
        .def("total_days", &TimeDelta::getTotalDays)
        .def("total_hours", &TimeDelta::getTotalHours)
        .def("total_minutes", &TimeDelta::getTotalMinutes)
        .def("total_seconds", &TimeDelta::getTotalSeconds)
        .def("__repr__", [](const TimeDelta & self)
                {
                    std::string s = "TimeDelta(";
                    s += "days=" + std::to_string(self.days) + ", ";
                    s += "hours=" + std::to_string(self.hours) + ", ";
                    s += "minutes=" + std::to_string(self.minutes) + ", ";
                    s += "seconds=" + std::to_string(self.seconds) + ", ";
                    s += "frac=" + std::to_string(self.frac) + ")";
                    return s;
                })
        ;
}
