#include "geocode.h"

#include "GeocodeSlc.h"
#include "GeocodeCov.h"

namespace py = pybind11;

void addsubmodule_geocode(py::module & m)
{
    py::module geocode = m.def_submodule("geocode");

    addbinding_geocodeslc(geocode);

    // forward declare bound classes
    py::class_<isce3::geocode::Geocode<float>>
        pyGeocodeFloat32(geocode, "GeocodeFloat32");
    py::class_<isce3::geocode::Geocode<double>>
        pyGeocodeFloat64(geocode, "GeocodeFloat64");
    py::class_<isce3::geocode::Geocode<std::complex<float>>>
        pyGeocodeCFloat32(geocode, "GeocodeCFloat32");
    py::class_<isce3::geocode::Geocode<std::complex<double>>>
        pyGeocodeCFloat64(geocode, "GeocodeCFloat64");

    // forward declare bound enums
    py::enum_<isce3::geocode::geocodeMemoryMode>
        pyGeocodeMemoryMode(geocode, "GeocodeMemoryMode");
    py::enum_<isce3::geocode::geocodeOutputMode>
        pyGeocodeOutputMode(geocode, "GeocodeOutputMode");

    // add bindings
    addbinding(pyGeocodeFloat32);
    addbinding(pyGeocodeFloat64);
    addbinding(pyGeocodeCFloat32);
    addbinding(pyGeocodeCFloat64);
    addbinding(pyGeocodeMemoryMode);
    addbinding(pyGeocodeOutputMode);


}
