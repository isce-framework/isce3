#include "cuda.h"

#include "core/core.h"
#include "focus/focus.h"
#include "geometry/geometry.h"

namespace py = pybind11;

void addsubmodule_cuda(py::module& m)
{
    py::module m_cuda = m.def_submodule("cuda");

    addsubmodule_cuda_core(m_cuda);
    addsubmodule_cuda_focus(m_cuda);
    addsubmodule_cuda_geometry(m_cuda);
}
