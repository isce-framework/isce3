#include "geocode.h"

#include "GeocodeSlc.h"

namespace py = pybind11;

void addsubmodule_geocode(py::module & m)
{
    py::module geocode = m.def_submodule("geocode");

    addbinding_geocodeslc(geocode);
}
