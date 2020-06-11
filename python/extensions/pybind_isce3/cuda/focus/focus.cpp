#include "focus.h"

#include "Backproject.h"

namespace py = pybind11;

void addsubmodule_cuda_focus(py::module& m)
{
    py::module m_focus = m.def_submodule("focus");

    // add bindings
    addbinding_cuda_backproject(m_focus);
}
