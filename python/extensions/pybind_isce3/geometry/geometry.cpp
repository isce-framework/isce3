#include "geometry.h"

#include "DEMInterpolator.h"
#include "RTC.h"
#include "boundingbox.h"
#include "geo2rdr.h"
#include "metadataCubes.h"
#include "rdr2geo.h"
#include "ltpcoordinates.h"
#include "pntintersect.h"

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
            pyInputTerrainRadiometry(geometry, "RtcInputTerrainRadiometry");
    py::enum_<isce3::geometry::rtcOutputTerrainRadiometry>
            pyOutputTerrainRadiometry(geometry, "RtcOutputTerrainRadiometry");
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
    addbinding(pyInputTerrainRadiometry);
    addbinding(pyOutputTerrainRadiometry);
    addbinding(pyRtcAlgorithm);
    addbinding(pyRtcMemoryMode);
    addbinding(pyRtcAreaMode);

    addbinding_apply_rtc(geometry);
    addbinding_compute_rtc(geometry);
    addbinding_compute_rtc_bbox(geometry);
    addbinding_geo2rdr(geometry);
    addbinding_rdr2geo(geometry);
    addbinding_boundingbox(geometry);
    addbinding_metadata_cubes(geometry);
    addbinding_ltp_coordinates(geometry);
    addbinding_pnt_intersect(geometry);
}
