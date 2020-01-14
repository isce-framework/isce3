#include "core.h"

#include "TimeDelta.h"

namespace py = pybind11;

void addsubmodule_core(py::module & m)
{
    py::module m_core = m.def_submodule("core");

    // forward declare bound classes
    py::class_<isce::core::TimeDelta> pyTimeDelta(m_core, "TimeDelta");

    // add bindings
    addbinding(pyTimeDelta);
}
