#pragma once
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/core/forward.h>

namespace isce3 { namespace geometry {


/** returns a DEM interpolator for a geocoded grid.
 * The geocoded grid and the input raster of the DEM can be in
 * different or same projection systems
 * @param[in] demRaster a raster of the DEM
 * @param[in] geoGrid  parameters of the geocoded grid
 * @param[in] demMarginInPixels  DEM extra margin in pixels
 * @param[in] demInterpMethod  DEM interpolation method
 */
DEMInterpolator DEMRasterToInterpolator(
        isce3::io::Raster& demRaster,
        const isce3::product::GeoGridParameters& geoGrid,
        const int demMarginInPixels = 50,
        const isce3::core::dataInterpMethod demInterpMethod =
            isce3::core::BIQUINTIC_METHOD);


/** returns a DEM interpolator for a block of geocoded grid.
 * The geocoded grid and the inpit raster of the DEM can be in
 * different or same projection systems
 * @param[in] demRaster a raster of the DEM
 * @param[in] geoGrid  parameters of the geocoded grid
 * @param[in] lineStart start line of the block of interest in the geocoded grid
 * @param[in] blockLength length of the block of interest in the geocoded grid
 * @param[in] blockWidth  width of the block of interest in the geocoded grid
 * @param[in] demMarginInPixels  DEM extra margin in pixels
 * @param[in] demInterpMethod  DEM interpolation method
 */
isce3::geometry::DEMInterpolator DEMRasterToInterpolator(
        isce3::io::Raster& demRaster,
        const isce3::product::GeoGridParameters& geoGrid, const int lineStart,
        const int blockLength, const int blockWidth,
        const int demMarginInPixels = 50,
        const isce3::core::dataInterpMethod demInterpMethod =
            isce3::core::BIQUINTIC_METHOD);


/** Load DEM raster into a DEMInterpolator object around a given bounding box
* in the same or different coordinate system as the DEM raster
*
* @param[in]  dem_raster              DEM raster
* @param[in]  x0                      Easting/longitude of western edge of bounding box,
* If the DEM is in geographic coordinates and the `x0` coordinate is not
* from the polar stereo system EPSG 3031 or EPSG 3413, this point represents
* the minimum X coordinate value. In this case, the maximum
* longitude span that this function can handle is 180 degrees
* (when the DEM is in geographic coordinates and `proj` is in polar stereo)
* @param[in]  xf                      Easting/longitude of eastern edge of bounding box
* If the DEM is in geographic coordinates and the `xf` coordinate is not
* from the polar stereo system EPSG 3031 or EPSG 3413, this point represents
* the maximum X coordinate value. In this case, the maximum
* longitude span that this function can handle is 180 degrees
* (when the DEM is in geographic coordinates and `proj` is in polar stereo)
* @param[in]  minY                    Minimum Y/northing position
* @param[in]  maxY                    Maximum Y/northing position
* @param[out] dem_interp_block        DEM interpolation object
* @param[in]  proj                    Projection object (nullptr to use same
* DEM projection)
* @param[in]  dem_margin_x_in_pixels  DEM X/easting margin in pixels
* @param[in]  dem_margin_y_in_pixels  DEM Y/northing margin in pixels
* @param[in]  dem_raster_band         DEM raster band (starting from 1)
*/
isce3::error::ErrorCode loadDemFromProj(
    isce3::io::Raster& dem_raster,
    const double minX, const double maxX, const double minY,
    const double maxY, isce3::geometry::DEMInterpolator* dem_interp,
    isce3::core::ProjectionBase* proj = nullptr,
    const int dem_margin_x_in_pixels = 100,
    const int dem_margin_y_in_pixels = 100,
    const int dem_raster_band = 1);

/*
 Interpolate DEM at position (x, y) considering that input_proj and
 dem_interp have same coordinate systems. The function is written to
 have the same interface of getDemCoordsDiffEpsg()
 * @param[in]  x           X-coordinate in input coordinates
 * @param[in]  y           Y-coordinate in input coordinates
 * @param[in]  dem_interp  DEM interpolation object
 * @param[in]  input_proj  Input projection object
 * @returns                3-elements vector containing the x and
 * y coordinates over DEM projection coordinates, and interpolated
 * DEM value at that position: {x_dem, y_dem, z_dem}
 */
isce3::core::Vec3 getDemCoordsSameEpsg(double x, double y,
        const DEMInterpolator& dem_interp,
        isce3::core::ProjectionBase* input_proj);

/*
 Convert x and y coordinates to from input_proj coordinates to
 DEM (dem_interp) coordinates and interpolate DEM at that position.
 3-elements vector containing the
 * @param[in]  x           X-coordinate in input coordinates
 * @param[in]  y           Y-coordinate in input coordinates
 * @param[in]  dem_interp  DEM interpolation object
 * @param[in]  input_proj  Input projection object
 * @returns                3-elements vector containing the x and
 * y coordinates over DEM projection coordinates, and interpolated
 * DEM value at that position: {x_dem, y_dem, z_dem}
 */
isce3::core::Vec3 getDemCoordsDiffEpsg(double x, double y,
        const DEMInterpolator& dem_interp,
        isce3::core::ProjectionBase* input_proj);


}} // namespace isce3::geometry
