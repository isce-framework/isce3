#pragma once
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>

#include <limits>

#include <pyre/journal.h>

#include <isce3/core/Common.h>
#include <isce3/core/Constants.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

#include "RadarGridParameters.h"

namespace isce3 { namespace product {

std::string to_string(const GeoGridParameters& geogrid);

std::ostream& operator<<(std::ostream& out, const GeoGridParameters& geogrid);

class GeoGridParameters {
public:
    GeoGridParameters() = default;

    /**
     * Construct new GeoGridParameters object from user defined values start,
     * spacing, and dimensions
     *
     * @param[in] geoGridStartX Start east-west/x position for geocoded grid
     * @param[in] geoGridStartY Start north/south y position for geocoded grid
     * @param[in] geoGridSpacingX East-west/x spacing for geocoded grid
     * @param[in] geoGridSpacingY North-south/y spacing for geocoded grid
     * @param[in] width Number of columns in geocoded grid (X direction)
     * @param[in] length Number of lines (rows) in geocoded grid (Y direction)
     * @param[in] epsgcode epsg code for geocoded grid
     */
    GeoGridParameters(double geoGridStartX, double geoGridStartY,
                             double geoGridSpacingX, double geoGridSpacingY,
                             int width, int length, int epsgcode);

    /**
     * Print GeoGridParameters attributes
     */
    void print() const;

    /** Set start x position for geocoded grid */
    void startX(double x0) { _startX = x0; }

    /** Set start y position for geocoded grid */
    void startY(double y0) { _startY = y0; }

    /** Set x spacing for geocoded grid */
    void spacingX(double dx) { _spacingX = dx; }

    /** Set y spacing for geocoded grid */
    void spacingY(double dy) { _spacingY = dy; }

    /** Set number of pixels in east-west/x direction for geocoded grid */
    void length(int l) { _length = l; };

    /** Set number of pixels in north-south/y direction for geocoded grid */
    void width(int w) { _width = w; };

    /** Set epsg code for geocoded grid */
    void epsg(int e) { _epsg = e; };

    /** Get start x position for geocoded grid */
    CUDA_HOSTDEV constexpr double startX() const { return _startX; };

    /** Get start y position for geocoded grid */
    CUDA_HOSTDEV constexpr double startY() const { return _startY; };

    /** Get end x position for geocoded grid */
    CUDA_HOSTDEV constexpr double endX() const {
        return _startX + _spacingX * _width;
    };

    /** Get end y position for geocoded grid */
    CUDA_HOSTDEV constexpr double endY() const {
        return _startY + _spacingY * _length;
    };

    /** Get x spacing for geocoded grid */
    CUDA_HOSTDEV constexpr double spacingX() const { return _spacingX; };

    /** Get y spacing for geocoded grid */
    CUDA_HOSTDEV constexpr double spacingY() const { return _spacingY; };

    /** Get number of pixels in east-west/x direction for geocoded grid */
    CUDA_HOSTDEV constexpr int width() const { return _width; };

    /** Get number of pixels in north-south/y direction for geocoded grid */
    CUDA_HOSTDEV constexpr int length() const { return _length; };

    /** Get epsg code for geocoded grid */
    CUDA_HOSTDEV constexpr int epsg() const { return _epsg; };


protected:
    /** start X position for the geocoded grid */
    double _startX = 0.0;

    /** start Y position for the geocoded grid */
    double _startY = 0.0;

    /** X spacing for the geocoded grid */
    double _spacingX = 0.0;

    /** Y spacing for the geocoded grid */
    double _spacingY = 0.0;

    /** number of pixels in east-west direction (X direction) */
    int _width = 0;

    /** number of lines in north-south direction (Y direction) */
    int _length = 0;

    /** epsg code for the output geocoded grid */
    int _epsg = 4326;
};

/**
 * Function to create a GeoGridParameters object by using DEM spacing and EPSG, and
 * by estimating the bounding box of the input radar grid.
 *
 * @param[in] radar_grid Input RadarGridParameters
 * @param[in] orbit Input orbit
 * @param[in] doppler Input doppler
 * @param[in] dx X spacing for geocoded grid
 * @param[in] dy Y spacing for geocoded grid
 * @param[in] epsg EPSG code
 * @param[in] min_height Height lower bound
 * @param[in] max_height Height upper bound
 * @param[in] margin Amount to pad estimated bounding box. In decimal degrees.
 * @param[in] point_per_edge Number of points to use on each side of radar grid.
 * @param[in] threshold Height threshold (m) for rdr2geo convergence.
 * @param[in] height_threshold Height threshold for convergence.
 */
GeoGridParameters
bbox2GeoGrid(const isce3::product::RadarGridParameters& radar_grid,
             const isce3::core::Orbit& orbit,
             const isce3::core::LUT2d<double>& doppler, double spacing_x,
             double spacing_y, int epsg,
             double min_height = isce3::core::GLOBAL_MIN_HEIGHT,
             double max_height = isce3::core::GLOBAL_MAX_HEIGHT,
             const double margin = 0.0, const int points_per_edge = 11,
             const double threshold = isce3::geometry::detail::DEFAULT_TOL_HEIGHT,
             const double height_threshold = 100);

/**
 * Function to create a GeoGridParameters object by using DEM spacing and EPSG, and
 * by estimating the bounding box of the input radar grid. Spacing can be adjusted by scalar.
 *
 * @param[in] radar_grid Input RadarGridParameters
 * @param[in] orbit Input orbit
 * @param[in] doppler Input doppler
 * @param[in] dem_raster DEM from which EPSG and spacing is extracted
 * @param[in] spacing_scale Scalar increase or decrease geogrid spacing
 * @param[in] min_height Height lower bound
 * @param[in] max_height Height upper bound
 * @param[in] margin Amount to pad estimated bounding box. In decimal degrees.
 * @param[in] point_per_edge Number of points to use on each side of radar grid.
 * @param[in] threshold Height threshold (m) for rdr2geo convergence.
 * @param[in] height_threshold Height threshold for convergence.
 */
GeoGridParameters
bbox2GeoGridScaled(const isce3::product::RadarGridParameters& radar_grid,
                   const isce3::core::Orbit& orbit,
                   const isce3::core::LUT2d<double>& doppler,
                   const isce3::io::Raster& dem_raster,
                   double spacing_scale = 1.0,
                   double min_height = isce3::core::GLOBAL_MIN_HEIGHT,
                   double max_height = isce3::core::GLOBAL_MAX_HEIGHT,
                   const double margin = 0.0, const int points_per_edge = 11,
                   const double threshold = isce3::geometry::detail::DEFAULT_TOL_HEIGHT,
                   const double height_threshold = 100);


}} // namespace isce3::product
