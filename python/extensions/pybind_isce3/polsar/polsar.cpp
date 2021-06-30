#include "polsar.h"

#include "symmetrize.h"

void addsubmodule_polsar(py::module& m)
{
    py::module m_polsar = m.def_submodule("polsar");

    addbinding_symmetrize(m_polsar);
}
