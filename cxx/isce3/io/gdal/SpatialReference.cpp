#include "SpatialReference.h"

#include <cstdlib>

#include <isce3/except/Error.h>

namespace isce { namespace io { namespace gdal {

static
std::string getOGRErrString(OGRErr status)
{
    switch (status) {
        case OGRERR_NONE                      : return "Success";
        case OGRERR_NOT_ENOUGH_DATA           : return "Not enough data to deserialize";
        case OGRERR_NOT_ENOUGH_MEMORY         : return "Not enough memory";
        case OGRERR_UNSUPPORTED_GEOMETRY_TYPE : return "Unsupported geometry type";
        case OGRERR_UNSUPPORTED_OPERATION     : return "Unsupported operation";
        case OGRERR_CORRUPT_DATA              : return "Corrupt data";
        case OGRERR_FAILURE                   : return "Failure";
        case OGRERR_UNSUPPORTED_SRS           : return "Unsupported SRS";
        case OGRERR_INVALID_HANDLE            : return "Invalid handle";
        case OGRERR_NON_EXISTING_FEATURE      : return "Non existing feature";
    }

    throw isce::except::RuntimeError(ISCE_SRCINFO(), "unknown OGRErr code");
}

SpatialReference::SpatialReference(int epsg)
{
    OGRErr status = _srs.importFromEPSG(epsg);
    if (status != OGRERR_NONE) {
        std::string errmsg =
            "failed to initialize spatial reference "
            "from EPSG code '" + std::to_string(epsg) + "' - " +
            getOGRErrString(status);
        throw isce::except::GDALError(ISCE_SRCINFO(), errmsg);
    }
}

SpatialReference::SpatialReference(const std::string & wkt)
{
    OGRErr status = _srs.importFromWkt(wkt.c_str());
    if (status != OGRERR_NONE) {
        std::string errmsg = "failed to initialize spatial reference from WKT string - " + getOGRErrString(status);
        throw isce::except::GDALError(ISCE_SRCINFO(), errmsg);
    }

    // add EPSG authority code if missing
    status = _srs.AutoIdentifyEPSG();
    if (status != OGRERR_NONE) {
        throw isce::except::GDALError(ISCE_SRCINFO(), "failed to identify EPSG authority code");
    }
}

SpatialReference::SpatialReference(const OGRSpatialReference & srs)
:
    _srs(srs)
{

    // add EPSG authority code if missing
    OGRErr status = _srs.AutoIdentifyEPSG();
    if (status != OGRERR_NONE) {
        throw isce::except::GDALError(ISCE_SRCINFO(), "failed to identify EPSG authority code");
    }
}

int SpatialReference::toEPSG() const
{
    // get EPSG code
    const char * code = _srs.GetAuthorityCode(nullptr);
    if (!code) {
        throw isce::except::GDALError(ISCE_SRCINFO(), "unable to fetch EPSG code");
    }
    int epsg = std::atoi(code);

    return epsg;
}

std::string SpatialReference::toWKT() const
{
    char * tmp = nullptr;
    OGRErr status = _srs.exportToPrettyWkt(&tmp);
    if (status != OGRERR_NONE) {
        std::string errmsg = "unable to export spatial reference to WKT - " + getOGRErrString(status);
        throw isce::except::GDALError(ISCE_SRCINFO(), errmsg);
    }
    std::string wkt = tmp;

    CPLFree(tmp);

    return wkt;
}

bool operator==(const SpatialReference & lhs, const SpatialReference & rhs)
{
    return lhs.get().IsSame( &(rhs.get()) );
}

bool operator!=(const SpatialReference & lhs, const SpatialReference & rhs)
{
    return !(lhs == rhs);
}

}}}
