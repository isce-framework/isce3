#include "signal.h"

#include "Crossmul.h"

namespace py = pybind11;

void addsubmodule_cuda_signal(py::module & m)
{
    py::module m_signal = m.def_submodule("signal");

    // forward declare bound classes
    py::class_<isce3::cuda::signal::gpuCrossmul> pyCrossmul(m_signal, "Crossmul");

    // add bindings
    addbinding(pyCrossmul);
}
