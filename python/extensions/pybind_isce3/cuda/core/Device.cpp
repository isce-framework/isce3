#include "Device.h"

#include <pybind11/operators.h>

namespace py = pybind11;

using isce3::cuda::core::ComputeCapability;
using isce3::cuda::core::Device;

void addbinding(py::class_<Device>& pyDevice)
{
    pyDevice
            // constructor(s)
            .def(py::init<int>(), py::arg("id"),
                 "Construct a new Device object.\n\nDoes not change the "
                 "currently active CUDA device.")

            // member access
            .def_property_readonly("id", &Device::id)
            .def_property_readonly("name", &Device::name)
            .def_property_readonly("total_global_mem", &Device::totalGlobalMem)
            .def_property_readonly("compute_capability",
                                   &Device::computeCapability)

            // magic methods
            .def("__repr__",
                 [](Device self) {
                     return "Device(id=" + std::to_string(self.id()) + ")";
                 })

            // operators
            .def(py::self == py::self)
            .def(py::self != py::self)
            ;
}
