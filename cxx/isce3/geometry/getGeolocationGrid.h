#pragma once

#include <isce3/core/Constants.h>
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>
#include <isce3/product/forward.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

namespace isce3 { namespace geometry {

/** Get geolocation grid from L1 products
 *
 * The target-to-sensor line-of-sight (LOS) and along-track unit vectors are
 * referenced to ENU coordinates computed wrt targets.
 *
 * @param[in]  dem_raster                  DEM raster
 * @param[in]  radar_grid                  Radar grid
 * @param[in]  orbit                       Reference orbit
 * @param[in]  native_doppler              Native image Doppler
 * @param[in]  grid_doppler                Grid Doppler
 * @param[in]  epsg                        Output geolocation EPSG
 * @param[in]  dem_interp_method           DEM interpolation method
 * @param[in]  rdr2geo_params              Geo2rdr parameters
 * @param[in]  geo2rdr_params              Rdr2geo parameters
 * @param[in]  threshold_geo2rdr           Range threshold for geo2rdr
 * @param[in]  numiter_geo2rdr             Geo2rdr maximum number of iterations
 * @param[in]  delta_range                 Step size used for computing
 * derivative of doppler
 * @param[out] interpolated_dem_raster     Interpolated DEM raster
 * @param[out] coordinate_x_raster         Coordinate-X raster
 * @param[out] coordinate_y_raster         Coordinate-Y raster
 * @param[out] incidence_angle_raster      Incidence angle (in degrees wrt 
 * ellipsoid normal at target) cube raster
 * @param[out] los_unit_vector_x_raster    LOS (target-to-sensor) unit vector
 * X cube raster
 * @param[out] los_unit_vector_y_raster    LOS (target-to-sensor) unit vector
 * Y cube raster
 * @param[out] along_track_unit_vector_x_raster Along-track unit vector X 
 * cube raster
 * @param[out] along_track_unit_vector_y_raster Along-track unit vector Y 
 * cube raster
 * @param[out] elevation_angle_raster      Elevation angle (in degrees wrt 
 * geodedic nadir) cube raster
 * @param[out] ground_track_velocity_raster Ground-track velocity raster
*/
void getGeolocationGrid(
        isce3::io::Raster& dem_raster,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& native_doppler,
        const isce3::core::LUT2d<double>& grid_doppler,
        const int epsg,
        isce3::core::dataInterpMethod dem_interp_method =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
        const isce3::geometry::detail::Rdr2GeoParams& rdr2geo_params = {},
        const isce3::geometry::detail::Geo2RdrParams& geo2rdr_params = {},
        isce3::io::Raster* interpolated_dem_raster = nullptr,
        isce3::io::Raster* coordinate_x_raster = nullptr,
        isce3::io::Raster* coordinate_y_raster = nullptr,
        isce3::io::Raster* incidence_angle_raster = nullptr,
        isce3::io::Raster* los_unit_vector_x_raster = nullptr,
        isce3::io::Raster* los_unit_vector_y_raster = nullptr,
        isce3::io::Raster* along_track_unit_vector_x_raster = nullptr,
        isce3::io::Raster* along_track_unit_vector_y_raster = nullptr,
        isce3::io::Raster* elevation_angle_raster = nullptr,
        isce3::io::Raster* ground_track_velocity_raster = nullptr
        );

}}