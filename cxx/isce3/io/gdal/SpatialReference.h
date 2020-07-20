#pragma once

#include <ogr_spatialref.h>
#include <string>

namespace isce3 { namespace io { namespace gdal {

/** Spatial reference system / coordinate reference system */
class SpatialReference {
public:

    /** Construct from EPSG code. */
    explicit SpatialReference(int epsg);

    /** Construct from WKT-formatted string. */
    explicit SpatialReference(const std::string & wkt);

    SpatialReference(const OGRSpatialReference &);

    /** Get EPSG code. */
    int toEPSG() const;

    /** Export to WKT-formatted string. */
    std::string toWKT() const;

    /** Get the underlying OGRSpatialReference object. */
    const OGRSpatialReference & get() const { return _srs; }

private:
    OGRSpatialReference _srs;
};

/** True if the two objects describe the same spatial reference system. */
bool operator==(const SpatialReference &, const SpatialReference &);

/** True if the two objects do not describe the same spatial reference system. */
bool operator!=(const SpatialReference &, const SpatialReference &);

}}}
