#include "GeoGridParameters.h"
#include <isce3/core/Projections.h>
#include <isce3/except/Error.h>
#include <isce3/geometry/boundingbox.h>
#include <isce3/io/Raster.h>

namespace isce3 { namespace product {

GeoGridParameters::GeoGridParameters(
        double geoGridStartX, double geoGridStartY, double geoGridSpacingX,
        double geoGridSpacingY, int width, int length, int epsgcode) :
      // the starting coordinate of the output geocoded grid in X direction.
      _startX(geoGridStartX),

      // the starting coordinate of the output geocoded grid in Y direction.
      _startY(geoGridStartY),

      // spacing of the output geocoded grid in X
      _spacingX(geoGridSpacingX),

      // spacing of the output geocoded grid in Y
      _spacingY(geoGridSpacingY),

      // number of lines (rows) in the geocoded grid (Y direction)
      _width(width),

      // number of columns in the geocoded grid (Y direction)
      _length(length),

      // Save the EPSG code
      _epsg(epsgcode)
{
    if (geoGridSpacingY >= 0.0) {
        std::string errmsg = "Y spacing can not be positive.";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }
}

std::string to_string(const GeoGridParameters& geogrid)
{
    std::string ret_str = "start X: " + std::to_string(geogrid.startX()) + "\n";
    ret_str += "end X: " +
               std::to_string(geogrid.startX() +
                              geogrid.spacingX() * geogrid.width()) +
               "\n";
    ret_str += "start Y: " + std::to_string(geogrid.startY()) + "\n";
    ret_str += "end Y: " +
               std::to_string(geogrid.startY() +
                              geogrid.spacingY() * geogrid.length()) +
               "\n";
    ret_str += "spacing X: " + std::to_string(geogrid.spacingX()) + "\n";
    ret_str += "spacing Y: " + std::to_string(geogrid.spacingY()) + "\n";
    ret_str += "width: " + std::to_string(geogrid.width()) + "\n";
    ret_str += "length: " + std::to_string(geogrid.length()) + "\n";
    ret_str += "epsg: " + std::to_string(geogrid.epsg()) + "\n";
    return ret_str;
}

std::ostream& operator<<(std::ostream& out, const GeoGridParameters& geogrid)
{
    return out << to_string(geogrid);
}

void GeoGridParameters::print() const
{

    pyre::journal::info_t info("isce.product.GeoGridParameters");

    std::string ret_str = to_string(*this);
    std::string delimiter = "\n";

    // remove last delimiter
    ret_str.erase(ret_str.length() - delimiter.length(), std::string::npos);

    // substitute "\n" by pyre::journal::newline
    size_t pos = 0;
    while ((pos = ret_str.find(delimiter)) != std::string::npos) {
        info << ret_str.substr(0, pos) << pyre::journal::newline;
        ret_str.erase(0, pos + delimiter.length());
    }
    info << ret_str << pyre::journal::endl;
}

GeoGridParameters bbox2GeoGrid(const RadarGridParameters& radar_grid,
                               const isce3::core::Orbit& orbit,
                               const isce3::core::LUT2d<double>& doppler,
                               double spacing_x, double spacing_y, int epsg,
                               double min_height, double max_height,
                               const double margin, const int pointsPerEdge,
                               const double threshold,
                               const double height_threshold)
{
    if (spacing_x <= 0) {
        std::string errmsg = "X spacing must be > 0.0";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }

    if (spacing_y >= 0.0) {
        std::string errmsg = "Y spacing must be < 0.0";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }


    // determine bounding box from radar grid
    auto proj = isce3::core::makeProjection(epsg);
    isce3::geometry::BoundingBox bbox = isce3::geometry::getGeoBoundingBoxHeightSearch(
            radar_grid, orbit,
            proj.get(), doppler,
            min_height, max_height,
            margin, pointsPerEdge,
            threshold,
            height_threshold);

    // retrieve geogrid values based on bounding box values
    auto start_x = bbox.MinX;
    auto start_y = bbox.MaxY;
    auto width = static_cast<int>(std::ceil((bbox.MaxX - bbox.MinX) / spacing_x));
    auto length = static_cast<int>(std::ceil(std::abs((bbox.MaxY - bbox.MinY) / spacing_y)));

    return GeoGridParameters(start_x, start_y, spacing_x, spacing_y, width, length, epsg);
}

GeoGridParameters bbox2GeoGridScaled(
        const RadarGridParameters& radar_grid, const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& doppler,
        const isce3::io::Raster& dem_raster, double spacing_scale, 
        double min_height, double max_height, const double margin,
        const int pointsPerEdge, const double threshold,
        const double height_threshold)
{
    if (spacing_scale <= 0.0) {
        std::string errmsg = "Spacing must be > 0.0";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }

    // retrieve spacing values from DEM raster and scale
    auto spacing_x = dem_raster.dx() * spacing_scale;
    auto spacing_y = dem_raster.dy() * spacing_scale;

    int epsg = dem_raster.getEPSG();

    return bbox2GeoGrid(radar_grid, orbit, doppler, spacing_x,
                        spacing_y, epsg, min_height, max_height, margin,
                        pointsPerEdge, threshold, height_threshold);
}
}}
