#include "geocode.h"

#include "Geocode.h"

namespace py = pybind11;

using isce3::cuda::geocode::Geocode;

void addsubmodule_cuda_geocode(py::module & m)
{
    py::module m_geocode = m.def_submodule("geocode");

    // forward declare bound classes
    py::class_<Geocode> pyGeocode(m_geocode, "Geocode");

    // add bindings
    addbinding(pyGeocode);
}
