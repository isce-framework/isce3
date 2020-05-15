#include "geometry.h"

#include "boundingbox.h"
#include "DEMInterpolator.h"
#include "rdr2geo.h"

namespace py = pybind11;

void addsubmodule_geometry(py::module & m)
{
    py::module geometry = m.def_submodule("geometry");

    // forward declare bound classes
    py::class_<isce::geometry::DEMInterpolator>
        pyDEMInterpolator(geometry, "DEMInterpolator");

    // add bindings
    addbinding(pyDEMInterpolator);
    addbinding_rdr2geo(geometry);
    addbinding_boundingbox(geometry);
}
