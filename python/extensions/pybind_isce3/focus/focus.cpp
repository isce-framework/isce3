#include "focus.h"

#include "Backproject.h"
#include "DryTroposphereModel.h"
#include "RangeComp.h"

namespace py = pybind11;

void addsubmodule_focus(py::module & m)
{
    py::module m_focus = m.def_submodule("focus");

    // forward declare bound enums
    py::enum_<isce::focus::DryTroposphereModel> pyDryTropoModel(m_focus, "DryTroposphereModel");

    py::class_<isce::focus::RangeComp> pyRangeComp(m_focus, "RangeComp");
    py::enum_<isce::focus::RangeComp::Mode> pyMode(pyRangeComp, "Mode");

    // add bindings
    addbinding(pyDryTropoModel);
    addbinding(pyMode);

    addbinding_backproject(m_focus);
    addbinding(pyRangeComp);
}
