#include "geometry.h"

#include "DEMInterpolator.h"
#include "getGeolocationGrid.h"
#include "RTC.h"
#include "boundingbox.h"
#include "geo2rdr.h"
#include "geo2rdr_roots.h"
#include "metadataCubes.h"
#include "rdr2geo.h"
#include "rdr2geo_roots.h"
#include "ltpcoordinates.h"
#include "pntintersect.h"
#include "lookIncFromSr.h"

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
    py::class_<isce3::geometry::RadarGridBoundingBox>
        pyRadarGridBoundingBox(geometry, "RadarGridBoundingBox");
    py::class_<isce3::geometry::detail::Geo2RdrParams>
        pyGeo2RdrParams(geometry, "Geo2RdrParams");
    py::class_<isce3::geometry::detail::Rdr2GeoParams>
        pyRdr2GeoParams(geometry, "Rdr2GeoParams");

    // forward declare bound enums
    py::enum_<isce3::geometry::rtcInputTerrainRadiometry>
            pyInputTerrainRadiometry(geometry, "RtcInputTerrainRadiometry");
    py::enum_<isce3::geometry::rtcOutputTerrainRadiometry>
            pyOutputTerrainRadiometry(geometry, "RtcOutputTerrainRadiometry");
    py::enum_<isce3::geometry::rtcAlgorithm>
        pyRtcAlgorithm(geometry, "RtcAlgorithm");
    py::enum_<isce3::geometry::rtcAreaMode>
        pyRtcAreaMode(geometry, "RtcAreaMode");
    py::enum_<isce3::geometry::rtcAreaBetaMode>
        pyRtcAreaBetaMode(geometry, "RtcAreaBetaMode");

    // add bindings
    addbinding(pyDEMInterpolator);
    addbinding(pyGeo2Rdr);
    addbinding(pyRdr2Geo);
    addbinding(pyInputTerrainRadiometry);
    addbinding(pyOutputTerrainRadiometry);
    addbinding(pyRtcAlgorithm);
    addbinding(pyRtcAreaMode);
    addbinding(pyRtcAreaBetaMode);
    addbinding(pyRadarGridBoundingBox);
    addbinding(pyGeo2RdrParams);
    addbinding(pyRdr2GeoParams);

    addbinding_apply_rtc(geometry);
    addbinding_compute_rtc(geometry);
    addbinding_compute_rtc_bbox(geometry);
    addbinding_get_geolocation_grid(geometry);
    addbinding_geo2rdr(geometry);
    addbinding_geo2rdr_roots(geometry);
    addbinding_rdr2geo(geometry);
    addbinding_rdr2geo_roots(geometry);
    addbinding_boundingbox(geometry);
    addbinding_metadata_cubes(geometry);
    addbinding_ltp_coordinates(geometry);
    addbinding_pnt_intersect(geometry);
    addbinding_look_inc_from_sr(geometry);
    addbinding_DEM_raster2interpolator(geometry);
}
