#include "ComputeCapability.h"

#include <pybind11/operators.h>

namespace py = pybind11;

using isce3::cuda::core::ComputeCapability;

void addbinding(py::class_<ComputeCapability>& pyComputeCapability)
{
    pyComputeCapability
            // constructor(s)
            .def(py::init<int, int>(), py::arg("major"), py::arg("minor"),
                 "Construct a new ComputeCapability object.")

            // member access
            .def_readwrite("major", &ComputeCapability::major)
            .def_readwrite("minor", &ComputeCapability::minor)

            // magic methods
            .def("__repr__",
                 [](ComputeCapability self) {
                     return "ComputeCapability(major=" +
                            std::to_string(self.major) +
                            ", minor=" + std::to_string(self.minor) + ")";
                 })
            .def("__str__", &ComputeCapability::operator std::string)

            // operators
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            ;
}
