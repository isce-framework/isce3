#include "cuda.h"

#include "focus/focus.h"

namespace py = pybind11;

void addsubmodule_cuda(py::module& m)
{
    py::module m_cuda = m.def_submodule("cuda");

    addsubmodule_cuda_focus(m_cuda);
}
