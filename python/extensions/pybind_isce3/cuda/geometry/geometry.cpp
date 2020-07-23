#include "geometry.h"

#include "geo2rdr.h"
#include "rdr2geo.h"

namespace py = pybind11;

void addsubmodule_cuda_geometry(py::module & m)
{
    py::module geometry = m.def_submodule("geometry");

    // forward declare bound classes
    py::class_<isce3::cuda::geometry::Geo2rdr>
        pyGeo2Rdr(geometry, "Geo2Rdr");
    py::class_<isce3::cuda::geometry::Topo>
        pyRdr2Geo(geometry, "Rdr2Geo");

    // add bindings
    addbinding(pyGeo2Rdr);
    addbinding(pyRdr2Geo);
}
