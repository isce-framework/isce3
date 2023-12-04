#include "DateTime.h"

#include <datetime.h>
#include <memory>
#include <pybind11/operators.h>
#include <string>

#include <isce3/core/TimeDelta.h>

namespace py = pybind11;

using isce3::core::DateTime;
using isce3::core::TimeDelta;

static
DateTime fromPyDateTime(py::handle obj)
{
    // initialize PyDateTime module
    if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

    // check that argument is not null and is instance of
    // datetime.datetime, datetime.date, or datetime.time
    if (!obj || !PyDateTime_Check(obj.ptr())) {
        PyErr_SetString(PyExc_TypeError, "invalid datetime object");
        throw py::error_already_set();
    }

    // XXX python C API doesn't provide a way to check tzinfo
    // XXX treat all input datetimes as naive & interpret as UTC time

    int year  = PyDateTime_GET_YEAR(obj.ptr());
    int month = PyDateTime_GET_MONTH(obj.ptr());
    int day   = PyDateTime_GET_DAY(obj.ptr());

    int hour        = PyDateTime_DATE_GET_HOUR(obj.ptr());
    int minute      = PyDateTime_DATE_GET_MINUTE(obj.ptr());
    int second      = PyDateTime_DATE_GET_SECOND(obj.ptr());
    int microsecond = PyDateTime_DATE_GET_MICROSECOND(obj.ptr());

    return {year, month, day, hour, minute, second, 1e-6 * microsecond};
}

void addbinding(py::class_<DateTime> & pyDateTime)
{
    pyDateTime
        .def(py::init<>())
        .def(py::init<DateTime>())
        .def(py::init<double>(), py::arg("ord"), "Construct from ordinal")
        .def(py::init<int, int, int>(),
                py::arg("year"),
                py::arg("month"),
                py::arg("day"))
        .def(py::init<int, int, int, int, int, int>(),
                py::arg("year"),
                py::arg("month"),
                py::arg("day"),
                py::arg("hour"),
                py::arg("minute"),
                py::arg("second"))
        .def(py::init<int, int, int, int, int, double>(),
                py::arg("year"),
                py::arg("month"),
                py::arg("day"),
                py::arg("hour"),
                py::arg("minute"),
                py::arg("second"))
        .def(py::init<int, int, int, int, int, int, double>(),
                py::arg("year"),
                py::arg("month"),
                py::arg("day"),
                py::arg("hour"),
                py::arg("minute"),
                py::arg("second"),
                py::arg("frac"))
        .def(py::init<const std::string &>(), "Construct from ISO-8601 formatted string")
        .def(py::init([](py::handle obj)
                {
                    return std::make_unique<DateTime>(fromPyDateTime(obj));
                }),
                "Construct from datetime.datetime. "
                "Timezone info is discarded and the object is interpreted as UTC time.")
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self += TimeDelta())
        .def(py::self -= TimeDelta())
        .def(py::self + TimeDelta())
        .def(py::self - TimeDelta())
        .def(py::self - py::self)
        .def_readwrite("year", &DateTime::year)
        .def_readwrite("month", &DateTime::months)
        .def_readwrite("day", &DateTime::days)
        .def_readwrite("hour", &DateTime::hours)
        .def_readwrite("minute", &DateTime::minutes)
        .def_readwrite("second", &DateTime::seconds)
        .def_readwrite("frac", &DateTime::frac)
        .def("is_close",
                (bool (DateTime::*)(const DateTime &, const TimeDelta &) const)&DateTime::isClose,
                py::arg("other"),
                py::arg("tol") = TimeDelta(isce3::core::TOL_SECONDS))
//        .def("day_of_year", &DateTime::dayOfYear)  // XXX not implemented
        .def("seconds_of_day", &DateTime::secondsOfDay)
//        .def("day_of_week", &DateTime::dayOfWeek)  // XXX not implemented
//        .def("toordinal", &DateTime::ordinal)  // XXX not implemented
        .def("isoformat", &DateTime::isoformat)
        .def("isoformat_usec", [](const DateTime& self) {return self.isoformat().substr(0, 26);})
        .def("__str__", [](const DateTime & self)  { return std::string(self); })
        .def("__repr__", [](const DateTime & self) { return std::string(self); })
        ;
}
