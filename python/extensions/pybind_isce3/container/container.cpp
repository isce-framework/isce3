#include "container.h"

#include "RadarGeometry.h"

namespace py = pybind11;

void addsubmodule_container(py::module & m)
{
    py::module m_container = m.def_submodule("container");

    // forward declare bound classes
    py::class_<isce::container::RadarGeometry> pyRadarGeometry(m_container, "RadarGeometry");

    // add bindings
    addbinding(pyRadarGeometry);
}
