#include "matchtemplate.h"

#include <isce3/cuda/matchtemplate/pycuampcor/cuAmpcorController.h>
#include <isce3/cuda/matchtemplate/pycuampcor/cuAmpcorParameter.h>

#include "pycuampcor.h"

namespace py = pybind11;

void addsubmodule_matchtemplate(py::module& m)
{
    py::module m_matchtemplate = m.def_submodule("matchtemplate");

    addbinding_pycuampcor_cpu(m_matchtemplate);
}
