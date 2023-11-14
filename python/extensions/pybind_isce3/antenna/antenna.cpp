#include "antenna.h"

#include "EdgeMethodCostFunc.h"
#include "ElNullRangeEst.h"
#include "ElPatternEst.h"
#include "Frame.h"
#include "SphGridType.h"
#include "geometryfunc.h"

namespace py = pybind11;

void addsubmodule_antenna(py::module& m)
{

    // main module
    py::module m_antenna = m.def_submodule("antenna");

    // declare structures
    py::class_<isce3::antenna::NullProduct> pyNullProduct(
            m_antenna, "NullProduct");

    py::class_<isce3::antenna::NullConvergenceFlags> pyNullConvergenceFlags(
            m_antenna, "NullConvergenceFlags");

    py::class_<isce3::antenna::NullPowPatterns> pyNullPowPatterns(
            m_antenna, "NullPowPatterns");

    // declare classes
    py::class_<isce3::antenna::Frame> pyFrame(m_antenna, "Frame");

    py::class_<isce3::antenna::ElPatternEst> pyElPatternEst(
            m_antenna, "ElPatternEst");

    py::class_<isce3::antenna::ElNullRangeEst> pyElNullRangeEst(
            m_antenna, "ElNullRangeEst");

    // declare enums
    py::enum_<isce3::antenna::SphGridType> pySphGridType(
            m_antenna, "SphGridType");

    // call addbinding for adding above pybind class/enums/struct
    addbinding(pyNullProduct);
    addbinding(pyNullConvergenceFlags);
    addbinding(pyNullPowPatterns);
    addbinding(pyFrame);
    addbinding(pyElPatternEst);
    addbinding(pyElNullRangeEst);
    addbinding(pySphGridType);

    // for modules with pure functions
    addbinding_geometryfunc(m_antenna);
    addbinding_edge_method_cost_func(m_antenna);
}
