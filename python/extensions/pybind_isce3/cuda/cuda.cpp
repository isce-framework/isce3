#include "cuda.h"

#include "core/core.h"
#include "focus/focus.h"
#include "geocode/geocode.h"
#include "geometry/geometry.h"
#include "image/image.h"
#include "matchtemplate/matchtemplate.h"
#include "signal/signal.h"

namespace py = pybind11;

void addsubmodule_cuda(py::module& m)
{
    py::module m_cuda = m.def_submodule("cuda");

    addsubmodule_cuda_core(m_cuda);
    addsubmodule_cuda_focus(m_cuda);
    addsubmodule_cuda_geocode(m_cuda);
    addsubmodule_cuda_geometry(m_cuda);
    addsubmodule_cuda_image(m_cuda);
    addsubmodule_cuda_matchtemplate(m_cuda);
    addsubmodule_cuda_signal(m_cuda);
}
