#include "Orbit.h"

#include <isce/core/Serialization.h>

#include <pybind11/chrono.h>
#include <pybind11/operators.h>

#include <string>

namespace py = pybind11;
using isce::core::Orbit;

#include <pybind11/eigen.h>

void addbinding(py::class_<Orbit> & pyOrbit)
{
    pyOrbit
        /*
         * XXX
         * This is an inefficient helper method to load
         * an orbit from an H5 file that isn't open yet.
         * If you're loading many objects from the same file,
         * don't use this pattern! Keep the file open!
         */
        .def_static("load_from_h5", [](std::string file_name,
                                       std::string group_name) {
            using namespace isce::io;

            // Get the H5 group
            IH5File h5file{file_name};
            auto igroup = h5file.openGroup(group_name);

            // Load the orbit
            Orbit o;
            isce::core::loadFromH5(igroup, o);
            return o;
        })

        // trivial member getters
        .def("spacing",        &Orbit::spacing)
        .def("size",           &Orbit::size)
        .def("start_time",     &Orbit::startTime)
        .def("end_time",       &Orbit::endTime)
        .def("start_datetime", &Orbit::startDateTime)
        .def("mid_datetime",   &Orbit::midDateTime)
        .def("end_datetime",   &Orbit::endDateTime)

        // trivial indexed getters
        .def(    "time_at", py::overload_cast<int>(&Orbit::time,     py::const_))
        .def("position_at", py::overload_cast<int>(&Orbit::position, py::const_))
        .def("velocity_at", py::overload_cast<int>(&Orbit::velocity, py::const_))
        ;
}
