#pragma once

#include <isce3/core/Constants.h>
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>
#include <isce3/product/forward.h>
#include <isce3/geometry/detail/Geo2Rdr.h>

namespace isce3 { namespace geogrid {

/** Get radar grid from L2 products
 *
 * Each output layer is saved onto the first band of its
 * associated raster file.
 * 
 * The line-of-sight (LOS) and along-track unit vectors are
 * referenced to ENU coordinates computed wrt targets.
 *
 * @param[in]  lookside                    Look side
 * @param[in]  wavelength                  Wavelength
 * @param[in]  dem_raster                  DEM raster
 * @param[in]  geogrid                     Output layers geogrid
 * @param[in]  orbit                       Reference orbit
 * @param[in]  native_doppler              Native image Doppler
 * @param[in]  grid_doppler                Grid Doppler
 * @param[in]  dem_interp_method           DEM interpolation method
 * @param[in]  geo2rdr_params              Geo2rdr parameters
 * @param[out] interpolated_dem_raster     Interpolated DEM raster
 * @param[out] slant_range_raster          Slant-range (in meters) 
 * cube raster
 * @param[out] azimuth_time_raster         Azimuth time (in seconds relative
 * to orbit epoch) cube raster
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
 * @param[out] local_incidence_angle_raster Local-incidence angle raster
 * @param[out] projection_angle_raster     Projection angle raster
 * @param[out] simulated_radar_brightness_raster Simulated radar brightness
 * raster
 */
void getRadarGrid(
        isce3::core::LookSide lookside,
        const double wavelength,
        isce3::io::Raster& dem_raster,
        const isce3::product::GeoGridParameters& geogrid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& native_doppler,
        const isce3::core::LUT2d<double>& grid_doppler,
        isce3::core::dataInterpMethod dem_interp_method =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD,
        const isce3::geometry::detail::Geo2RdrParams& geo2rdr_params = {},
        isce3::io::Raster* interpolated_dem_raster = nullptr,
        isce3::io::Raster* slant_range_raster = nullptr,
        isce3::io::Raster* azimuth_time_raster = nullptr,
        isce3::io::Raster* incidence_angle_raster = nullptr,
        isce3::io::Raster* los_unit_vector_x_raster = nullptr,
        isce3::io::Raster* los_unit_vector_y_raster = nullptr,
        isce3::io::Raster* along_track_unit_vector_x_raster = nullptr,
        isce3::io::Raster* along_track_unit_vector_y_raster = nullptr,
        isce3::io::Raster* elevation_angle_raster = nullptr,
        isce3::io::Raster* ground_track_velocity_raster = nullptr,
        isce3::io::Raster* local_incidence_angle_raster = nullptr,
        isce3::io::Raster* projection_angle_raster = nullptr,
        isce3::io::Raster* simulated_radar_brightness_raster = nullptr);

}}