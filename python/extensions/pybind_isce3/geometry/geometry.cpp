#include "geometry.h"

#include "boundingbox.h"
#include "DEMInterpolator.h"
#include "geo2rdr.h"
#include "rdr2geo.h"
#include "RTC.h"
#include "metadataCubes.h"

namespace py = pybind11;

void addsubmodule_geometry(py::module & m)
{
    py::module geometry = m.def_submodule("geometry");

    // forward declare bound classes
    py::class_<isce3::geometry::DEMInterpolator>
        pyDEMInterpolator(geometry, "DEMInterpolator");
    py::class_<isce3::geometry::Geo2rdr>
        pyGeo2Rdr(geometry, "Geo2Rdr");
    py::class_<isce3::geometry::Topo>
        pyRdr2Geo(geometry, "Rdr2Geo");

    // forward declare bound enums
    py::enum_<isce3::geometry::rtcInputTerrainRadiometry>
        pyinputTerrainRadiometry(geometry, "RtcInputTerrainRadiometry");
    py::enum_<isce3::geometry::rtcAlgorithm>
        pyRtcAlgorithm(geometry, "RtcAlgorithm");
    py::enum_<isce3::geometry::rtcMemoryMode>
        pyRtcMemoryMode(geometry, "RtcMemoryMode");
    py::enum_<isce3::geometry::rtcAreaMode>
        pyRtcAreaMode(geometry, "RtcAreaMode");

    // add bindings
    addbinding(pyDEMInterpolator);
    addbinding(pyGeo2Rdr);
    addbinding(pyRdr2Geo);
    addbinding(pyinputTerrainRadiometry);
    addbinding(pyRtcAlgorithm);
    addbinding(pyRtcMemoryMode);
    addbinding(pyRtcAreaMode);

    addbinding_apply_rtc(geometry);
    addbinding_facet_rtc(geometry);
    addbinding_facet_rtc_bbox(geometry);
    addbinding_geo2rdr(geometry);
    addbinding_rdr2geo(geometry);
    addbinding_boundingbox(geometry);
    addbinding_metadata_cubes(geometry);
}
