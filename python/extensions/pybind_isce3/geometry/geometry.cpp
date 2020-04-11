#include "geometry.h"

#include "DEMInterpolator.h"

namespace py = pybind11;

void addsubmodule_geometry(py::module & m)
{
    py::module geometry = m.def_submodule("geometry");

    // forward declare bound classes
    py::class_<isce::geometry::DEMInterpolator>
        pyDEMInterpolator(geometry, "DEMInterpolator");

    // add bindings
    addbinding(pyDEMInterpolator);
}
