#pragma once

#include <isce3/core/LUT2d.h>

//isce3::product
#include <isce3/product/RadarGridParameters.h>
#include <isce3/product/GeoGridParameters.h>
//isce3::geometry
#include "Shapes.h"
#include "DEMInterpolator.h"
#include "detail/Geo2Rdr.h"
#include "detail/Rdr2Geo.h"

//Declaration
namespace isce3{ namespace geometry{

/* Light container defining the indices of a radar grid bounding box */
struct RadarGridBoundingBox {
    int firstAzimuthLine;
    int lastAzimuthLine;
    int firstRangeSample;
    int lastRangeSample;
};

/** Compute the perimeter of a radar grid in map coordinates.
 *
 * @param[in] radarGrid     RadarGridParameters object
 * @param[in] orbit         Orbit object
 * @param[in] proj          ProjectionBase object indicating desired
 *                          map projection of output.
 * @param[in] doppler       LUT2d doppler model
 * @param[in] demInterp     DEM Interpolator
 * @param[in] pointsPerEge  Number of points to use on each edge of radar grid
 * @param[in] threshold     Height threshold (m) for rdr2geo convergence
 *
 * The output of this method is an OGRLinearRing.
 *
 * The sequence of walking the perimeter is guaranteed to have counter-clockwise
 * order on a map projection, which is required by the ISO 19125 specification
 * for Simple Features used by OGR/GDAL.  (Note that ESRI shapefiles use the
 * opposite convention.)
 *
 * A consequence of the consistently CCW map ordering is that the radar
 * coordinate traversal order depends on the look direction.  If we label the
 * corners as follows:
 *      1. Early time, near range
 *      2. Early time, far range
 *      3. Late time, far range
 *      4. Late time, near range
 * then in the right-looking case the edges follow a 12341 ordering.  For the
 * left-looking case, the edges follow a 14321 ordering instead.
 */
Perimeter getGeoPerimeter(const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit,
        const isce3::core::ProjectionBase* proj,
        const isce3::core::LUT2d<double>& doppler = {},
        const DEMInterpolator& demInterp = DEMInterpolator(0.),
        const int pointsPerEdge = 11,
        const double threshold = detail::DEFAULT_TOL_HEIGHT);


/** Compute bounding box using min/ max altitude for quick estimates
 *
 * @param[in] radarGrid    RadarGridParameters object
 * @param[in] orbit         Orbit object
 * @param[in] proj          ProjectionBase object indicating desired projection
 * of output.
 * @param[in] doppler       LUT2d doppler model
 * @param[in] hgts          Vector of heights to use for the scene
 * @param[in] margin        Marging to add to estimated bounding box in decimal
 * degrees
 * @param[in] pointsPerEge  Number of points to use on each edge of radar grid
 * @param[in] threshold     Height threshold (m) for rdr2geo convergence
 *
 * The output of this method is an OGREnvelope.
 */
BoundingBox getGeoBoundingBox(
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit,
        const isce3::core::ProjectionBase* proj,
        const isce3::core::LUT2d<double>& doppler = {},
        const std::vector<double>& hgts = {isce3::core::GLOBAL_MIN_HEIGHT,
                isce3::core::GLOBAL_MAX_HEIGHT},
        const double margin = 0.0, const int pointsPerEdge = 11,
        const double threshold = detail::DEFAULT_TOL_HEIGHT,
        bool ignore_out_of_range_exception = false);

/** Compute bounding box with auto search within given min/ max height
 * interval
 *
 * @param[in] radarGrid    RadarGridParameters object
 * @param[in] orbit         Orbit object
 * @param[in] proj          ProjectionBase object indicating desired
 * projection of output.
 * @param[in] doppler       LUT2d doppler model
 * @param[in] minHeight     Height lower bound
 * @param[in] maxHeight     Height upper bound
 * @param[in] margin        Margin to add to estimated bounding box in
 * decimal degrees
 * @param[in] pointsPerEge  Number of points to use on each edge of radar
 * grid
 * @param[in] threshold     Height threshold (m) for rdr2geo convergence
 * @param[in] height_threshold Height threshold for convergence
 * The output of this method is an OGREnvelope.
 */
BoundingBox getGeoBoundingBoxHeightSearch(
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit,
        const isce3::core::ProjectionBase* proj,
        const isce3::core::LUT2d<double>& doppler = {},
        double min_height = isce3::core::GLOBAL_MIN_HEIGHT,
        double max_height = isce3::core::GLOBAL_MAX_HEIGHT,
        const double margin = 0.0, const int pointsPerEdge = 11,
        const double threshold = detail::DEFAULT_TOL_HEIGHT,
        const double height_threshold = 100);

/** Compute bounding box of a geocoded grid within radar grid.
 *
 * The output of this function is a RadarGridBoundingBox object that defines
 * the bounding box that is extended by the input margin in all directions. The
 * output object contains index of:
 * 1) the first line in azimuth
 * 2) the last line in azimuth
 * 3) first pixel in range
 * 4) the last pixel in range
 * An exception is raised if:
 * 1) any corner fails to converge
 * 2) computed bounding box min index >= max index along an axis
 * 3) computed bounding box index is out of bounds
 *
 * @param[in] geoGrid       GeoGridParameters object
 * @param[in] radarGrid     RadarGridParameters object
 * @param[in] orbit         Orbit object
 * @param[in] min_height    minimum height value to be used with x and y
 *                          corners for geo2rdr computations
 * @param[in] max_height    maximum height value to be used with x and y
 *                          corners for geo2rdr computations
 * @param[in] doppler       LUT2d doppler model of the radar image grid
 * @param[in] margin        Margin to add to estimated bounding box in
 * pixels (default 50)
 * @param[in] g2r_params    Stucture containing the following geo2rdr params:
 *      azimuth time threshold for convergence (default: 1e-8)
 *      maximum number of iterations for convergence (default: 50)
 *      step size used for computing derivative of doppler (default: 10.0)
 * @param[in] geogrid_expansion_threshold       Maximum number of iterations to
 * outwardly expand each geogrid corner to search for geo2rdr convergence. An
 * outward extension shifts a corner by (geogrid.dx, geogrid.dy) (default: 100)
 * @returns                 RadarGridBoundingBox object defines radar grid
 *      bouding box - contains indices of first_azimuth_line,
 *      last_azimuth_line, first_range_sample, last_range_sample
 */
RadarGridBoundingBox getRadarBoundingBox(
        const isce3::product::GeoGridParameters& geoGrid,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit,
        const float min_height = isce3::core::GLOBAL_MIN_HEIGHT,
        const float max_height = isce3::core::GLOBAL_MAX_HEIGHT,
        const isce3::core::LUT2d<double>& doppler = {},
        const int margin = 50,
        const isce3::geometry::detail::Geo2RdrParams& g2r_params = {},
        const int geogrid_expansion_threshold = 100);

/** Compute bounding box of a geocoded grid within radar grid. Use this
 * for one off radar grid bounding box computations. Repeated use of this
 * function with the same DEM wastefully recomputes min and max DEM height.
 *
 * The output of this function is a RadarGridBoundingBox object that defines
 * the bounding box that is extended by the input margin in all directions. The
 * output object contains index of:
 * 1) the first line in azimuth
 * 2) the last line in azimuth
 * 3) first pixel in range
 * 4) the last pixel in range
 * An exception is raised if:
 * 1) any corner fails to converge
 * 2) computed bounding box min index >= max index along an axis
 * 3) computed bounding box index is out of bounds
 *
 * @param[in] geoGrid       GeoGridParameters object
 * @param[in] radarGrid     RadarGridParameters object
 * @param[in] orbit         Orbit object
 * @param[in] demInterp     DEMInterpolator object used to find min and max
 *                          height values to used in geo2rdr computation.
 * @param[in] doppler       LUT2d doppler model of the radar image grid
 * @param[in] margin        Margin to add to estimated bounding box in
 * pixels (default 50)
 * @param[in] g2r_params    Stucture containing the following geo2rdr params:
 *      azimuth time threshold for convergence (default: 1e-8)
 *      maximum number of iterations for convergence (default: 50)
 *      step size used for computing derivative of doppler (default: 10.0)
 * @param[in] geogrid_expansion_threshold       Maximum number of iterations to
 * outwardly expand each geogrid corner to search for geo2rdr convergence. An
 * outward extension shifts a corner by (geogrid.dx, geogrid.dy) (default: 100)
 * @returns                 RadarGridBoundingBox object defines radar grid
 *      bouding box - contains indices of first_azimuth_line,
 *      last_azimuth_line, first_range_sample, last_range_sample
 */
RadarGridBoundingBox getRadarBoundingBox(
        const isce3::product::GeoGridParameters& geoGrid,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit,
        isce3::geometry::DEMInterpolator& dem_interp,
        const isce3::core::LUT2d<double>& doppler = {},
        const int margin = 50,
        const isce3::geometry::detail::Geo2RdrParams& g2r_params = {},
        const int geogrid_expansion_threshold = 100);

}} // namespace isce3::geometry
