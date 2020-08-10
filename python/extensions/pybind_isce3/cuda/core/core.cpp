#include "core.h"

#include "ComputeCapability.h"
#include "Device.h"

namespace py = pybind11;

void addsubmodule_cuda_core(py::module& m)
{
    py::module m_core = m.def_submodule("core");

    // forward declare bound classes
    py::class_<isce3::cuda::core::ComputeCapability> pyComputeCapability(
            m_core, "ComputeCapability");
    py::class_<isce3::cuda::core::Device> pyDevice(m_core, "Device");

    // add bindings
    addbinding(pyComputeCapability);
    addbinding(pyDevice);

    m_core.def("min_compute_capability",
               &isce3::cuda::core::minComputeCapability);
    m_core.def("get_device_count", &isce3::cuda::core::getDeviceCount,
               "Return the number of available CUDA devices.");
    m_core.def("get_device", &isce3::cuda::core::getDevice,
               "Get the current CUDA device for the active host thread.");
    m_core.def("set_device", &isce3::cuda::core::setDevice, py::arg("device"),
               "Set the CUDA device for the active host thread.");
    m_core.def(
            "set_device", [](int d) { return isce3::cuda::core::setDevice(d); },
            py::arg("device"),
            "Set the CUDA device for the active host thread.");
}
