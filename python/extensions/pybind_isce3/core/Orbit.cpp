#include "Orbit.h"

#include <isce/core/Serialization.h>
#include <isce/core/Vector.h>

#include <pybind11/chrono.h>
#include <pybind11/operators.h>

#include <string>

namespace py = pybind11;
using isce::core::Orbit;

#include <pybind11/eigen.h>

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
        .def_property_readonly("position", [](const Orbit & self) {
            return py::array{toBuffer(self.position()), py::cast(self)};
        })
        .def_property_readonly("velocity", [](const Orbit & self) {
            return py::array{toBuffer(self.velocity()), py::cast(self)};
        })

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
        .def_property_readonly("spacing",        &Orbit::spacing)
        .def_property_readonly("size",           &Orbit::size)
        .def_property_readonly("start_time",     &Orbit::startTime)
        .def_property_readonly("end_time",       &Orbit::endTime)
        .def_property_readonly("start_datetime", &Orbit::startDateTime)
        .def_property_readonly("mid_datetime",   &Orbit::midDateTime)
        .def_property_readonly("end_datetime",   &Orbit::endDateTime)
        ;
}
