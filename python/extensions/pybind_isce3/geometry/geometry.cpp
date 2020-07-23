#include "geometry.h"

#include "boundingbox.h"
#include "DEMInterpolator.h"
#include "Geocode.h"
#include "geo2rdr.h"
#include "rdr2geo.h"
#include "RTC.h"

namespace py = pybind11;

void addsubmodule_geometry(py::module & m)
{
    py::module geometry = m.def_submodule("geometry");

    // forward declare bound classes
    py::class_<isce3::geometry::DEMInterpolator>
        pyDEMInterpolator(geometry, "DEMInterpolator");
    py::class_<isce3::geometry::Geocode<float>>
        pyGeocodeFloat32(geometry, "GeocodeFloat32");
    py::class_<isce3::geometry::Geocode<double>>
        pyGeocodeFloat64(geometry, "GeocodeFloat64");
    py::class_<isce3::geometry::Geocode<std::complex<float>>>
        pyGeocodeCFloat32(geometry, "GeocodeCFloat32");
    py::class_<isce3::geometry::Geocode<std::complex<double>>>
        pyGeocodeCFloat64(geometry, "GeocodeCFloat64");
    py::class_<isce3::geometry::Geo2rdr>
        pyGeo2Rdr(geometry, "Geo2Rdr");
    py::class_<isce3::geometry::Topo>
        pyRdr2Geo(geometry, "Rdr2Geo");

    // forward declare bound enums
    py::enum_<isce3::geometry::geocodeMemoryMode>
        pyGeocodeMemoryMode(geometry, "GeocodeMemoryMode");
    py::enum_<isce3::geometry::geocodeOutputMode>
        pyGeocodeOutputMode(geometry, "GeocodeOutputMode");
    py::enum_<isce3::geometry::rtcInputRadiometry>
        pyInputRadiometry(geometry, "RtcInputRadiometry");
    py::enum_<isce3::geometry::rtcAlgorithm>
        pyRtcAlgorithm(geometry, "RtcAlgorithm");

    // add bindings
    addbinding(pyDEMInterpolator);
    addbinding(pyGeocodeFloat32);
    addbinding(pyGeocodeFloat64);
    addbinding(pyGeocodeCFloat32);
    addbinding(pyGeocodeCFloat64);
    addbinding(pyGeo2Rdr);
    addbinding(pyRdr2Geo);
    addbinding(pyGeocodeMemoryMode);
    addbinding(pyGeocodeOutputMode);
    addbinding(pyInputRadiometry);
    addbinding(pyRtcAlgorithm);
    addbinding_rdr2geo(geometry);
    addbinding_boundingbox(geometry);
}
