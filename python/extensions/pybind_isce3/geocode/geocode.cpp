#include "geocode.h"

#include "GeocodeCov.h"
#include "GeocodePolygon.h"
#include "GeocodeSlc.h"

namespace py = pybind11;

void addsubmodule_geocode(py::module & m)
{
    py::module geocode = m.def_submodule("geocode");

    addbinding_geocodeslc<isce3::core::LUT2d<double>>(geocode);
    addbinding_geocodeslc<isce3::core::Poly2d>(geocode);

    // forward declare bound classes
    py::class_<isce3::geocode::Geocode<float>>
        pyGeocodeFloat32(geocode, "GeocodeFloat32");
    py::class_<isce3::geocode::Geocode<double>>
        pyGeocodeFloat64(geocode, "GeocodeFloat64");
    py::class_<isce3::geocode::Geocode<std::complex<float>>>
        pyGeocodeCFloat32(geocode, "GeocodeCFloat32");
    py::class_<isce3::geocode::Geocode<std::complex<double>>>
        pyGeocodeCFloat64(geocode, "GeocodeCFloat64");

    py::class_<isce3::geocode::GeocodePolygon<float>>
        pyGeocodePolygonFloat32(geocode, "GeocodePolygonFloat32");
    py::class_<isce3::geocode::GeocodePolygon<double>>
        pyGeocodePolygonFloat64(geocode, "GeocodePolygonFloat64");
    py::class_<isce3::geocode::GeocodePolygon<std::complex<float>>>
        pyGeocodePolygonCFloat32(geocode, "GeocodePolygonCFloat32");
    py::class_<isce3::geocode::GeocodePolygon<std::complex<double>>>
        pyGeocodePolygonCFloat64(geocode, "GeocodePolygonCFloat64");

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

    addbinding(pyGeocodePolygonFloat32);
    addbinding(pyGeocodePolygonFloat64);
    addbinding(pyGeocodePolygonCFloat32);
    addbinding(pyGeocodePolygonCFloat64);

    addbinding(pyGeocodeMemoryMode);
    addbinding(pyGeocodeOutputMode);

}
