#include "antenna.h"

#include "Frame.h"
#include "SphGridType.h"
#include "geometryfunc.h"
#include "ElPatternEst.h"

namespace py = pybind11;

void addsubmodule_antenna(py::module & m)
{

    // main module
    py::module m_antenna = m.def_submodule("antenna");

    // declare classes
    py::class_<isce3::antenna::Frame> pyFrame(m_antenna, "Frame");

    py::class_<isce3::antenna::ElPatternEst> pyElPatternEst(m_antenna, "ElPatternEst");

    // declare enums
    py::enum_<isce3::antenna::SphGridType> pySphGridType(m_antenna, "SphGridType");

    // call addbinding for adding above pybind class/enums
    addbinding(pyFrame);
    addbinding(pyElPatternEst);
    addbinding(pySphGridType);

    // for modules with pure functions
    addbinding_geometryfunc(m_antenna);
}
