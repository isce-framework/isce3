#include "focus.h"

#include "Backproject.h"
#include "DryTroposphereModel.h"

namespace py = pybind11;

void addsubmodule_focus(py::module & m)
{
    py::module m_focus = m.def_submodule("focus");

    // forward declare bound enums
    py::enum_<isce::focus::DryTroposphereModel> pyDryTropoModel(m_focus, "DryTroposphereModel");

    // add bindings
    addbinding(pyDryTropoModel);

    addbinding_backproject(m_focus);
}
