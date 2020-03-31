#include "core.h"

#include "DateTime.h"
#include "Ellipsoid.h"
#include "LookSide.h"
#include "LUT1d.h"
#include "LUT2d.h"
#include "Orbit.h"
#include "TimeDelta.h"

namespace py = pybind11;

void addsubmodule_core(py::module & m)
{
    py::module m_core = m.def_submodule("core");

    // forward declare bound classes
    py::class_<isce::core::DateTime> pyDateTime(m_core, "DateTime");
    py::class_<isce::core::Ellipsoid> pyEllipsoid(m_core, "Ellipsoid");
    py::class_<isce::core::LUT1d<double>> pyLUT1d(m_core, "LUT1d");
    py::class_<isce::core::LUT2d<double>> pyLUT2d(m_core, "LUT2d");
    py::class_<isce::core::Orbit> pyOrbit(m_core, "Orbit");
    py::class_<isce::core::TimeDelta> pyTimeDelta(m_core, "TimeDelta");

    // forward declare bound enums
    py::enum_<isce::core::LookSide> pyLookSide(m_core, "LookSide");

    // add bindings
    addbinding(pyDateTime);
    addbinding(pyEllipsoid);
    addbinding(pyLookSide);
    addbinding(pyLUT1d);
    addbinding(pyLUT2d);
    addbinding(pyOrbit);
    addbinding(pyTimeDelta);
}
