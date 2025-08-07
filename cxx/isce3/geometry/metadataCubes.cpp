#include "metadataCubes.h"

#include <algorithm>
#include <iostream>

#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Matrix.h>
#include <isce3/core/Projections.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/io/Raster.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/geometry/geometry.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/product/RadarGridParameters.h>
#include <isce3/geometry/metadataCubes.h>

namespace isce3 {
namespace geometry {


template<class T>
static isce3::core::Matrix<T>
getNanArray(isce3::io::Raster* raster,
            const isce3::product::GeoGridParameters& geogrid)
{
    /*
    This function allocates memory for an array (`data_array`)
    using `geogrid` dimensions if an output raster (`raster`)
    is provided, i.e, if `raster`
    is not a null pointer `nullptr`.
    */
    isce3::core::Matrix<T> data_array;
    if (raster != nullptr) {
        data_array.resize(geogrid.length(), geogrid.width());
    }
    data_array.fill(std::numeric_limits<T>::quiet_NaN());
    return data_array;
}

template<class T>
static isce3::core::Matrix<T>
getNanArrayRadarGrid(isce3::io::Raster* raster,
              const isce3::product::RadarGridParameters& radar_grid)
{
    /*
    This function allocates memory for an array (`data_array`)
    using `radar_grid` dimensions if an output raster (`raster`)
    is provided, i.e, if `raster` is not a null pointer `nullptr`.
    */
    isce3::core::Matrix<T> data_array;
    if (raster != nullptr) {            
        data_array.resize(radar_grid.length(), radar_grid.width());
    }
    data_array.fill(std::numeric_limits<T>::quiet_NaN());
    return data_array;
}

template<class T>
static void writeArray(isce3::io::Raster* raster,
        isce3::core::Matrix<T>& data_array, int height_count)
{
    if (raster == nullptr) {
        return;
    }
#pragma omp critical
    {
        raster->setBlock(data_array.data(), 0, 0, data_array.width(),
                         data_array.length(), height_count + 1);
    }
}

void writeVectorDerivedCubes(const int array_pos_i,
        const int array_pos_j, const double native_azimuth_time,
        const isce3::core::Vec3& target_llh,
        const isce3::core::Orbit& orbit,
        const isce3::core::Ellipsoid& ellipsoid,
        isce3::io::Raster* incidence_angle_raster,
        isce3::core::Matrix<float>& incidence_angle_array,
        isce3::io::Raster* los_unit_vector_x_raster,
        isce3::core::Matrix<float>& los_unit_vector_x_array,
        isce3::io::Raster* los_unit_vector_y_raster,
        isce3::core::Matrix<float>& los_unit_vector_y_array,
        isce3::io::Raster* along_track_unit_vector_x_raster,
        isce3::core::Matrix<float>& along_track_unit_vector_x_array,
        isce3::io::Raster* along_track_unit_vector_y_raster,
        isce3::core::Matrix<float>& along_track_unit_vector_y_array,
        isce3::io::Raster* elevation_angle_raster,
        isce3::core::Matrix<float>& elevation_angle_array,
        isce3::io::Raster* ground_track_velocity_raster,
        isce3::core::Matrix<double>& ground_track_velocity_array,
        isce3::io::Raster* local_incidence_angle_raster,
        isce3::core::Matrix<float>& local_incidence_angle_array,
        isce3::io::Raster* projection_angle_raster,
        isce3::core::Matrix<float>& projection_angle_array,
        isce3::io::Raster* simulated_radar_brightness_raster,
        isce3::core::Matrix<float>& simulated_radar_brightness_array,
        isce3::core::Vec3* terrain_normal_unit_vec_enu,
        isce3::core::LookSide* lookside)
{

    const int i = array_pos_i;
    const int j = array_pos_j;

    /*
    Interpolate orbit at native_azimuth_time to compute look
    and velocity vectors in ECEF.
    */
    isce3::core::cartesian_t sat_xyz, vel_xyz;
    isce3::error::ErrorCode status =
            orbit.interpolate(&sat_xyz, &vel_xyz, native_azimuth_time,
                              isce3::core::OrbitInterpBorderMode::FillNaN);

    // If interpolation fails, skip
    if (status != isce3::error::ErrorCode::Success) {
        return;
    }

    // Get target position in ECEF (target_xyz)
    const isce3::core::Vec3 target_xyz = ellipsoid.lonLatToXyz(target_llh);

    // Ground-track velocity
    if (ground_track_velocity_raster != nullptr) {
        // cosine law: c^2 = a^2 + b^2 - 2.a.b.cos(AB)
        // cos(AB) = (a^2 + b^2 - c^2) / 2.a.b
        const double slant_range = (target_xyz - sat_xyz).norm();
        const double radius_target = target_xyz.norm();
        const double radius_platform = sat_xyz.norm();
        const double cos_alpha = ((radius_target * radius_target +
                                   radius_platform * radius_platform -
                                   slant_range * slant_range) /
                                  (2 * radius_target * radius_platform));

        const double ground_velocity =
            cos_alpha * radius_target * vel_xyz.norm() / radius_platform;
        ground_track_velocity_array(i, j) = ground_velocity;
    }

    // Create target-to-sat vector in ECEF
    const isce3::core::Vec3 look_vector_xyz =
            (sat_xyz - target_xyz).normalized();

    // Compute elevation angle calculated in ENU (geodedic)
    if (elevation_angle_raster != nullptr) {

        // Get platform position in llh (sat_llh)
        const isce3::core::Vec3 sat_llh = ellipsoid.xyzToLonLat(sat_xyz);

        // Get target-to-sat vector in ENU around the platform
        const isce3::core::Mat3 xyz2enu_sat =
                isce3::core::Mat3::xyzToEnu(sat_llh[1], sat_llh[0]);
        const isce3::core::Vec3 look_vector_enu_sat =
                xyz2enu_sat.dot(look_vector_xyz).normalized();
        const double cos_elevation = look_vector_enu_sat[2];
        elevation_angle_array(i, j) = std::acos(cos_elevation) * 180.0 / M_PI;

    }

    // Get target-to-sat vector in ENU around the target
    const isce3::core::Mat3 xyz2enu =
            isce3::core::Mat3::xyzToEnu(target_llh[1], target_llh[0]);
    const isce3::core::Vec3 look_vector_enu =
            xyz2enu.dot(look_vector_xyz).normalized();

    // Compute incidence angle in ENU (geodetic)
    if (incidence_angle_raster != nullptr) {
        const double cos_inc = std::abs(look_vector_enu[2]);
        incidence_angle_array(i, j) = std::acos(cos_inc) * 180.0 / M_PI;
    }

    // Check if terrain_normal_unit_vec_enu is required
    if ((local_incidence_angle_raster != nullptr ||
         projection_angle_raster != nullptr ||
         simulated_radar_brightness_raster != nullptr) &&
            terrain_normal_unit_vec_enu == nullptr) {
        std::string error_message = "ERROR terrain normal unit vector not";
        error_message += " provided to compute local-incidence angle,";
        error_message += " projection angle, and/or simulated radar";
        error_message += " brightness";
        throw isce3::except::RuntimeError(
            ISCE_SRCINFO(), error_message);
    }

    // Check if lookside is required
    if ((projection_angle_raster != nullptr ||
         simulated_radar_brightness_raster != nullptr) &&
            lookside == nullptr) {
        std::string error_message = "ERROR look side not";
        error_message += " provided to compute the projection angle";
        error_message += " and/or the simulated radar brightness";
        throw isce3::except::RuntimeError(
            ISCE_SRCINFO(), error_message);
    }

    double cos_theta_i = std::numeric_limits<double>::quiet_NaN();
    // Compute local-incidence angle in ENU (geodetic)
    if (local_incidence_angle_raster != nullptr) {
        cos_theta_i = (look_vector_enu).dot(
            *terrain_normal_unit_vec_enu);
        local_incidence_angle_array(i, j) = (std::acos(cos_theta_i) * 
                                             180.0 / M_PI);
    }

    double cos_psi = std::numeric_limits<double>::quiet_NaN();

    // Compute projection angle in ENU (geodetic)
    if (projection_angle_raster != nullptr) {

        // Compute velocity vector (ENU)
        const isce3::core::Vec3 vel_enu = xyz2enu.dot(vel_xyz);

        // Calculate psi angle between image plane and local slope
        Vec3 image_normal_unit_vec_enu =
            ((-look_vector_enu).cross(vel_enu)).normalized();
    
        if (*lookside == isce3::core::LookSide::Left) {
            image_normal_unit_vec_enu *= -1.0;
        }

        cos_psi = (*terrain_normal_unit_vec_enu).dot(
            image_normal_unit_vec_enu);

        projection_angle_array(i, j) = (std::acos(cos_psi) * 
                                             180.0 / M_PI);
    }

    // Compute simulated radar brightness
    if (simulated_radar_brightness_raster != nullptr &&
            cos_theta_i < 0) {
        simulated_radar_brightness_array(i, j) = 0;
    }
    else if (simulated_radar_brightness_raster != nullptr) {

        float simulated_radar_brightness =
            cos_theta_i / std::abs(cos_psi);

        simulated_radar_brightness_array(i, j) = \
            simulated_radar_brightness;
    }

    // Compute vectors in ENU coordinates around target

    // LOS unit vector X (ENU)
    if (los_unit_vector_x_raster != nullptr) {
        los_unit_vector_x_array(i, j) = look_vector_enu[0];
    }

    // LOS unit vector Y (ENU)
    if (los_unit_vector_y_raster != nullptr) {
        los_unit_vector_y_array(i, j) = look_vector_enu[1];
    }

    // If along_track_unit_vector is not needed, skip
    if (along_track_unit_vector_x_raster == nullptr &&
        along_track_unit_vector_y_raster == nullptr) {
        return;
    }

    // Compute velocity vector (ENU)
    const isce3::core::Vec3 along_track_vector =
            xyz2enu.dot(vel_xyz);
    const double horizontal_norm = std::sqrt(
        std::pow(along_track_vector[0], 2) +
        std::pow(along_track_vector[1], 2));

    /// Along-track unit vector X along the ground track without the vertical
    // component
    if (along_track_unit_vector_x_raster != nullptr) {
        along_track_unit_vector_x_array(i, j) =
            along_track_vector[0] / horizontal_norm;
    }

    // Along-track unit vector Y along the ground track without the vertical
    // component
    if (along_track_unit_vector_y_raster != nullptr) {
        along_track_unit_vector_y_array(i, j) =
            along_track_vector[1] / horizontal_norm;
    }

}

void makeRadarGridCubes(const isce3::product::RadarGridParameters& radar_grid,
        const isce3::product::GeoGridParameters& geogrid,
        const std::vector<double>& heights, const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& native_doppler,
        const isce3::core::LUT2d<double>& grid_doppler,
        isce3::io::Raster* slant_range_raster,
        isce3::io::Raster* azimuth_time_raster,
        isce3::io::Raster* incidence_angle_raster,
        isce3::io::Raster* los_unit_vector_x_raster,
        isce3::io::Raster* los_unit_vector_y_raster,
        isce3::io::Raster* along_track_unit_vector_x_raster,
        isce3::io::Raster* along_track_unit_vector_y_raster,
        isce3::io::Raster* elevation_angle_raster,
        isce3::io::Raster* ground_track_velocity_raster,
        const double threshold_geo2rdr, const int numiter_geo2rdr,
        const double delta_range, bool flag_set_output_rasters_geolocation,
        const bool flag_ground_velocity_from_rdr2geo)
{

    pyre::journal::info_t info("isce.geometry.makeRadarGridCubes");
    info << "cube height: " << heights.size() << pyre::journal::newline;
    info << "cube length: " << geogrid.length() << pyre::journal::newline;
    info << "cube width: " << geogrid.width() << pyre::journal::endl;
    info << "EPSG: " << geogrid.epsg() << pyre::journal::endl;

    geogrid.print();

    isce3::io::Raster * local_incidence_angle_raster = nullptr;
    isce3::core::Matrix<float> local_incidence_angle_array;
    isce3::io::Raster * projection_angle_raster = nullptr;
    isce3::core::Matrix<float> projection_angle_array;
    isce3::io::Raster * simulated_radar_brightness_raster = nullptr;
    isce3::core::Matrix<float> simulated_radar_brightness_array;
    isce3::core::Vec3* terrain_normal_vector = nullptr;
    isce3::core::LookSide* lookside = nullptr;

#pragma omp parallel for
    for (int height_count = 0; height_count < heights.size(); ++height_count) {

        auto proj = isce3::core::makeProjection(geogrid.epsg());
        double azimuth_time = radar_grid.sensingMid();
        double native_azimuth_time = radar_grid.sensingMid();
        double slant_range = radar_grid.midRange();
        double native_slant_range = radar_grid.midRange();
        auto height = heights[height_count];

        const isce3::core::Ellipsoid& ellipsoid = proj->ellipsoid();
        auto slant_range_array =
                getNanArray<double>(slant_range_raster, geogrid);
        auto azimuth_time_array =
                getNanArray<double>(azimuth_time_raster, geogrid);
        auto incidence_angle_array =
                getNanArray<float>(incidence_angle_raster, geogrid);
        auto los_unit_vector_x_array =
                getNanArray<float>(los_unit_vector_x_raster, geogrid);
        auto los_unit_vector_y_array =
                getNanArray<float>(los_unit_vector_y_raster, geogrid);
        auto along_track_unit_vector_x_array =
                getNanArray<float>(along_track_unit_vector_x_raster, geogrid);
        auto along_track_unit_vector_y_array =
                getNanArray<float>(along_track_unit_vector_y_raster, geogrid);
        auto elevation_angle_array =
                getNanArray<float>(elevation_angle_raster, geogrid);
        auto ground_track_velocity_array =
                getNanArray<double>(ground_track_velocity_raster, geogrid);

        /*
        The function `isce3::geometry::writeVectorDerivedCubes()` computes
        ground-track velocity using a theoretical expression. It determines
        whether to perform this computation based on the output ground-track
        velocity raster being `nullptr` or not.

        If the `flag_ground_velocity_from_rdr2geo` flag is enabled, we intend
        to compute the ground-track velocity using the rdr2geo method
        instead of the theoretical formula. In this case, the ground-track
        velocity raster passed to `writeVectorDerivedCubes()` should be
        `nullptr`.

        On the other hand, if `flag_ground_velocity_from_rdr2geo` is disabled,
        we want the theoretical method to be used, so the output ground-track
        velocity raster should be set to `ground_track_velocity_raster`.
        */
        isce3::io::Raster* ground_track_velocity_theoretical_raster = nullptr;

        if (!flag_ground_velocity_from_rdr2geo) {
            ground_track_velocity_theoretical_raster = ground_track_velocity_raster;
        }
        isce3::geometry::DEMInterpolator dem_interpolator(height, geogrid.epsg());

        for (int i = 0; i < geogrid.length(); ++i) {
            double pos_y = geogrid.startY() + (0.5 + i) * geogrid.spacingY();
            for (int j = 0; j < geogrid.width(); ++j) {
                double pos_x =
                        geogrid.startX() + (0.5 + j) * geogrid.spacingX();

                // Get target coordinates in the output projection system
                const isce3::core::Vec3 target_proj {pos_x, pos_y, height};

                // Get target coordinates in llh
                const isce3::core::Vec3 target_llh = proj->inverse(target_proj);

                // Get grid Doppler azimuth and slant-range position
                int converged = isce3::geometry::geo2rdr(
                        target_llh, ellipsoid, orbit, grid_doppler,
                        azimuth_time, slant_range, radar_grid.wavelength(),
                        radar_grid.lookSide(), threshold_geo2rdr,
                        numiter_geo2rdr, delta_range);

                // Check convergence
                if (!converged) {
                    azimuth_time = radar_grid.sensingMid();
                    slant_range = radar_grid.midRange();
                    continue;
                }

                // save grid Doppler slant-range position
                if (slant_range_raster != nullptr) {
                    slant_range_array(i, j) = slant_range;
                }

                // Save grid Doppler azimuth position
                if (azimuth_time_raster != nullptr) {
                    azimuth_time_array(i, j) = azimuth_time;
                }

                // If nothing else to save, skip
                if (incidence_angle_raster == nullptr &&
                    los_unit_vector_x_raster == nullptr &&
                    los_unit_vector_y_raster == nullptr &&
                    along_track_unit_vector_x_raster == nullptr &&
                    along_track_unit_vector_y_raster == nullptr &&
                    elevation_angle_raster == nullptr &&
                    ground_track_velocity_raster == nullptr) {
                    continue;
                }

                /*
                To retrieve platform position (considering
                native Doppler), estimate native_azimuth_time
                */
                converged = isce3::geometry::geo2rdr(
                        target_llh, ellipsoid, orbit, native_doppler,
                        native_azimuth_time, native_slant_range,
                        radar_grid.wavelength(), radar_grid.lookSide(),
                        threshold_geo2rdr, numiter_geo2rdr, delta_range);

                // Check convergence
                if (!converged) {
                    native_azimuth_time = radar_grid.sensingMid();
                    native_slant_range = radar_grid.midRange();
                    continue;
                }

                // Ground-track velocity using rdr2geo
                if (ground_track_velocity_raster != nullptr &&
                        flag_ground_velocity_from_rdr2geo) {

                        /*
                        Get target position (target_llh) considering grid Doppler
                        */
                        double fd = grid_doppler.eval(azimuth_time, slant_range);
                        
                        Vec3 target_llh_before, target_llh_after;
                        target_llh_before[2] = height;
                        target_llh_after[2] = height;

                        /*
                        Compute ground-track velocity using finite differences
                        where dt is the pulse-repetition interval (PRI), inverse
                        of the pulse-repetition frequency (PRF)
                        */
                        double dt = 1.0 / radar_grid.prf();
                        double azimuth_time_before = azimuth_time - dt / 2.0;
                        double azimuth_time_after = azimuth_time + dt / 2.0;

                        auto converged_before =
                                rdr2geo(azimuth_time_before, slant_range, fd, orbit, ellipsoid,
                                        dem_interpolator, target_llh_before,
                                        radar_grid.wavelength(),
                                        radar_grid.lookSide(), threshold_geo2rdr,
                                        numiter_geo2rdr, delta_range);

                        auto converged_after =
                                rdr2geo(azimuth_time_after, slant_range, fd, orbit, ellipsoid,
                                        dem_interpolator, target_llh_after,
                                        radar_grid.wavelength(),
                                        radar_grid.lookSide(), threshold_geo2rdr,
                                        numiter_geo2rdr, delta_range);

                        /*
                        Since we already have the central point, we need at least one extra point
                        that converged to a solution
                        */
                        if (converged_before || converged_after) {
                            /*
                            If the previous point didn't converge, replace it by the central point
                            */
                            if (!converged_before) {
                                dt = dt / 2.0;
                                target_llh_before = target_llh;
                            }

                            /*
                            Otherwise, if the next point didn't converge, replace it by the central
                            point
                            */
                            else if (!converged_after && converged_before) {
                                dt = dt / 2.0;
                                target_llh_after = target_llh;
                            }

                            /*
                            We compute the distance between `target_xyz_after` and
                            `target_xyz_before` and divide it by the associated time interval.
                            This distance does not consider the Earth's curvature
                            which should be in the order of 3mm for a 1km distance.
                            */
                            const isce3::core::Vec3 target_xyz_before = \
                                ellipsoid.lonLatToXyz(target_llh_before);
                            const isce3::core::Vec3 target_xyz_after = \
                                ellipsoid.lonLatToXyz(target_llh_after);

                            // derivative X wrt az. time
                            double dx_dt = (target_xyz_after[0] - target_xyz_before[0]) / dt;

                            // derivative Y wrt az. time
                            double dy_dt = (target_xyz_after[1] - target_xyz_before[1]) / dt;


                            // derivative Z wrt az. time
                            double dz_dt = (target_xyz_after[2] - target_xyz_before[2]) / dt;

                            ground_track_velocity_array(i, j) = std::sqrt(std::pow(dx_dt, 2) +
                                                                          std::pow(dy_dt, 2) +
                                                                          std::pow(dz_dt, 2));

                        }
                        else {
                            ground_track_velocity_array(i, j) = std::numeric_limits<double>::quiet_NaN();
                        }

                }

                isce3::geometry::writeVectorDerivedCubes(i, j,
                        native_azimuth_time, target_llh, orbit, ellipsoid,
                        incidence_angle_raster, incidence_angle_array,
                        los_unit_vector_x_raster, los_unit_vector_x_array,
                        los_unit_vector_y_raster, los_unit_vector_y_array,
                        along_track_unit_vector_x_raster,
                        along_track_unit_vector_x_array,
                        along_track_unit_vector_y_raster,
                        along_track_unit_vector_y_array, 
                        elevation_angle_raster,
                        elevation_angle_array,
                        ground_track_velocity_theoretical_raster,
                        ground_track_velocity_array,
                        local_incidence_angle_raster,
                        local_incidence_angle_array,
                        projection_angle_raster,
                        projection_angle_array,
                        simulated_radar_brightness_raster,
                        simulated_radar_brightness_array,
                        terrain_normal_vector, lookside);
            }
        }

        writeArray(slant_range_raster, slant_range_array, height_count);
        writeArray(azimuth_time_raster, azimuth_time_array, height_count);
        writeArray(incidence_angle_raster, incidence_angle_array, height_count);
        writeArray(los_unit_vector_x_raster, los_unit_vector_x_array,
                   height_count);
        writeArray(los_unit_vector_y_raster, los_unit_vector_y_array,
                   height_count);
        writeArray(along_track_unit_vector_x_raster,
                   along_track_unit_vector_x_array, height_count);
        writeArray(along_track_unit_vector_y_raster,
                   along_track_unit_vector_y_array, height_count);
        writeArray(elevation_angle_raster, elevation_angle_array, height_count);
        writeArray(ground_track_velocity_raster, ground_track_velocity_array,
                   height_count);
    }

    if (!flag_set_output_rasters_geolocation) {
        return;
    }


    double geotransform[] = {
            geogrid.startX(),  geogrid.spacingX(), 0, geogrid.startY(), 0,
            geogrid.spacingY()};

    if (slant_range_raster != nullptr) {
        slant_range_raster->setGeoTransform(geotransform);
        slant_range_raster->setEPSG(geogrid.epsg());
    }
    if (azimuth_time_raster != nullptr) {
        azimuth_time_raster->setGeoTransform(geotransform);
        azimuth_time_raster->setEPSG(geogrid.epsg());
    }
    if (incidence_angle_raster != nullptr) {
        incidence_angle_raster->setGeoTransform(geotransform);
        incidence_angle_raster->setEPSG(geogrid.epsg());
    }
    if (los_unit_vector_x_raster != nullptr) {
        los_unit_vector_x_raster->setGeoTransform(geotransform);
        los_unit_vector_x_raster->setEPSG(geogrid.epsg());
    }
    if (los_unit_vector_y_raster != nullptr) {
        los_unit_vector_y_raster->setGeoTransform(geotransform);
        los_unit_vector_y_raster->setEPSG(geogrid.epsg());
    }
    if (along_track_unit_vector_x_raster != nullptr) {
        along_track_unit_vector_x_raster->setGeoTransform(geotransform);
        along_track_unit_vector_x_raster->setEPSG(geogrid.epsg());
    }
    if (along_track_unit_vector_y_raster != nullptr) {
        along_track_unit_vector_y_raster->setGeoTransform(geotransform);
        along_track_unit_vector_y_raster->setEPSG(geogrid.epsg());
    }
    if (elevation_angle_raster != nullptr) {
        elevation_angle_raster->setGeoTransform(geotransform);
        elevation_angle_raster->setEPSG(geogrid.epsg());
    }
    if (ground_track_velocity_raster != nullptr) {
        ground_track_velocity_raster->setGeoTransform(geotransform);
        ground_track_velocity_raster->setEPSG(geogrid.epsg());
    }
}

void makeGeolocationGridCubes(
        const isce3::product::RadarGridParameters& radar_grid,
        const std::vector<double>& heights, const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& native_doppler,
        const isce3::core::LUT2d<double>& grid_doppler, const int epsg,
        isce3::io::Raster* coordinate_x_raster,
        isce3::io::Raster* coordinate_y_raster,
        isce3::io::Raster* incidence_angle_raster,
        isce3::io::Raster* los_unit_vector_x_raster,
        isce3::io::Raster* los_unit_vector_y_raster,
        isce3::io::Raster* along_track_unit_vector_x_raster,
        isce3::io::Raster* along_track_unit_vector_y_raster,
        isce3::io::Raster* elevation_angle_raster,
        isce3::io::Raster* ground_track_velocity_raster,
        const double threshold_geo2rdr, const int numiter_geo2rdr,
        const double delta_range, const bool flag_ground_velocity_from_rdr2geo)
{

    pyre::journal::info_t info("isce.geometry.makeGeolocationGridCubes");
    info << "cube height: " << heights.size() << pyre::journal::endl;

    info << "cube length: " << radar_grid.length() << pyre::journal::newline;
    info << "cube width: " << radar_grid.width() << pyre::journal::endl;
    info << "EPSG: " << epsg << pyre::journal::endl;

    isce3::io::Raster * local_incidence_angle_raster = nullptr;
    isce3::core::Matrix<float> local_incidence_angle_array;
    isce3::io::Raster * projection_angle_raster = nullptr;
    isce3::core::Matrix<float> projection_angle_array;
    isce3::io::Raster * simulated_radar_brightness_raster = nullptr;
    isce3::core::Matrix<float> simulated_radar_brightness_array;
    isce3::core::Vec3* terrain_normal_vector = nullptr;
    isce3::core::LookSide* lookside = nullptr;

    /*
    The function `isce3::geometry::writeVectorDerivedCubes()` computes
    ground-track velocity using a theoretical expression. It determines
    whether to perform this computation based on the output ground-track
    velocity raster being `nullptr` or not.

    If the `flag_ground_velocity_from_rdr2geo` flag is enabled, we intend
    to compute the ground-track velocity using the rdr2geo method
    instead of the theoretical formula. In this case, the ground-track
    velocity raster passed to `writeVectorDerivedCubes()` should be
    `nullptr`.

    On the other hand, if `flag_ground_velocity_from_rdr2geo` is disabled,
    we want the theoretical method to be used, so the output ground-track
    velocity raster should be set to `ground_track_velocity_raster`.
    */
    isce3::io::Raster* ground_track_velocity_theoretical_raster = nullptr;

    if (!flag_ground_velocity_from_rdr2geo) {
        ground_track_velocity_theoretical_raster = ground_track_velocity_raster;
    }

    #pragma omp parallel for
    for (int height_count = 0; height_count < heights.size(); ++height_count) {

        auto proj = isce3::core::makeProjection(epsg);
        const isce3::core::Ellipsoid& ellipsoid = proj->ellipsoid();

        auto coordinate_x_array = 
                getNanArrayRadarGrid<double>(coordinate_x_raster, radar_grid);
        auto coordinate_y_array =
                getNanArrayRadarGrid<double>(coordinate_y_raster, radar_grid);
        auto incidence_angle_array =
                getNanArrayRadarGrid<float>(incidence_angle_raster, radar_grid);
        auto los_unit_vector_x_array =
                getNanArrayRadarGrid<float>(los_unit_vector_x_raster, radar_grid);
        auto los_unit_vector_y_array =
                getNanArrayRadarGrid<float>(los_unit_vector_y_raster, radar_grid);
        auto along_track_unit_vector_x_array =
                getNanArrayRadarGrid<float>(along_track_unit_vector_x_raster, radar_grid);
        auto along_track_unit_vector_y_array =
                getNanArrayRadarGrid<float>(along_track_unit_vector_y_raster, radar_grid);
        auto elevation_angle_array =
                getNanArrayRadarGrid<float>(elevation_angle_raster, radar_grid);
        auto ground_track_velocity_array =
                getNanArrayRadarGrid<double>(ground_track_velocity_raster, radar_grid);

        isce3::core::Matrix<double> target_pos_x, target_pos_y, target_pos_z;
        /*
        Set the minimum (`i_0`) and maximum (`i_f') azimuth lines to process.
        The algorithm to compute the ground-track velocity from `rdr2geo`
        requires one extra azimuth line at the beginning and another at the end
        of the radar grid. Therefore, if the flag
        `flag_ground_velocity_from_rdr2geo` is enabled, subtract `1` from
        `i_0` and add `1` to `i_f`.
        */
        int i_0 = 0;
        int i_f = radar_grid.length() - 1;

        if (flag_ground_velocity_from_rdr2geo) {
            i_0 -= 1;
            i_f += 1;

            /*
            the target_pos arrays will have an extra line at the beginning and 
            at the end, therefore we add `2` to their length.
            */
            target_pos_x.resize(radar_grid.length() + 2, radar_grid.width());
            target_pos_y.resize(radar_grid.length() + 2, radar_grid.width());
            target_pos_z.resize(radar_grid.length() + 2, radar_grid.width());
            target_pos_x.fill(std::numeric_limits<double>::quiet_NaN());
            target_pos_y.fill(std::numeric_limits<double>::quiet_NaN());
            target_pos_z.fill(std::numeric_limits<double>::quiet_NaN());
        }

        auto height = heights[height_count];
        isce3::geometry::DEMInterpolator dem_interpolator(height, epsg);
        double native_azimuth_time = radar_grid.sensingMid();
        double native_slant_range = radar_grid.midRange();

        for (int i = i_0; i <= i_f; ++i) {
            double az_time = radar_grid.sensingTime(i);
            for (int j = 0; j < radar_grid.width(); ++j) {
                double slant_range = radar_grid.slantRange(j);
                Vec3 target_llh;
                /*
                Skip processing for radar grid points outside grid doppler
                */
                if (!grid_doppler.contains(az_time, slant_range)) {
                    continue;
                }

                /*
                Get target position (target_llh) considering grid Doppler
                */
                double fd = grid_doppler.eval(az_time, slant_range);
                target_llh[2] = height;

                auto converged =
                        rdr2geo(az_time, slant_range, fd, orbit, ellipsoid,
                                dem_interpolator, target_llh,
                                radar_grid.wavelength(),
                                radar_grid.lookSide(), threshold_geo2rdr,
                                numiter_geo2rdr, delta_range);

                // Check convergence
                if (!converged) {
                    continue;
                }

                if (flag_ground_velocity_from_rdr2geo) {
                    const isce3::core::Vec3 target_xyz = ellipsoid.lonLatToXyz(target_llh);

                    target_pos_x(i + 1, j) = target_xyz[0];
                    target_pos_y(i + 1, j) = target_xyz[1];
                    target_pos_z(i + 1, j) = target_xyz[2];

                }

                /*
                The extra azimuth lines are only used to populate
                the `target_pos_x`, `target_pos_y`, and `target_pos_z`
                arrays. Once they are populated, we don't need
                those extra lines anymore.
                */
                if (i < 0 or i > radar_grid.length() - 1) {
                    continue;
                }

                // Get target position in the output proj system
                isce3::core::Vec3 target_proj = proj->forward(target_llh);

                if (coordinate_x_raster != nullptr) {
                    coordinate_x_array(i, j) = target_proj[0];
                }
                if (coordinate_y_raster != nullptr) {
                    coordinate_y_array(i, j) = target_proj[1];
                }

                // If nothing else to save, skip
                if (incidence_angle_raster == nullptr &&
                        los_unit_vector_x_raster == nullptr &&
                        los_unit_vector_y_raster == nullptr &&
                        along_track_unit_vector_x_raster == nullptr &&
                        along_track_unit_vector_y_raster == nullptr &&
                        elevation_angle_raster == nullptr &&
                        ground_track_velocity_raster == nullptr) {
                    continue;
                }

                /*
                To retrieve platform position (considering
                native Doppler), estimate native_azimuth_time 
                */
                converged = geo2rdr(target_llh, ellipsoid, orbit, native_doppler,
                        native_azimuth_time, native_slant_range,
                        radar_grid.wavelength(), radar_grid.lookSide(),
                        threshold_geo2rdr, numiter_geo2rdr, delta_range);

                // Check convergence
                if (!converged) {
                    native_azimuth_time = radar_grid.sensingMid();
                    native_slant_range = radar_grid.midRange();
                    continue;
                }

                writeVectorDerivedCubes(i, j, native_azimuth_time, target_llh,
                        orbit, ellipsoid,
                        incidence_angle_raster, incidence_angle_array,
                        los_unit_vector_x_raster, los_unit_vector_x_array,
                        los_unit_vector_y_raster, los_unit_vector_y_array,
                        along_track_unit_vector_x_raster,
                        along_track_unit_vector_x_array,
                        along_track_unit_vector_y_raster,
                        along_track_unit_vector_y_array, 
                        elevation_angle_raster,
                        elevation_angle_array,
                        ground_track_velocity_theoretical_raster,
                        ground_track_velocity_array,
                        local_incidence_angle_raster,
                        local_incidence_angle_array,
                        projection_angle_raster,
                        projection_angle_array,
                        simulated_radar_brightness_raster,
                        simulated_radar_brightness_array,
                        terrain_normal_vector, lookside);
            }
        }

        // Ground-track velocity
        if (ground_track_velocity_raster != nullptr &&
                flag_ground_velocity_from_rdr2geo) {

            if (height_count == 0) {
                info << "estimating the ground-track velocity using rdr2geo" << pyre::journal::endl;
            }

            _Pragma("omp parallel for")
            /* 
            Compute ground-track velocity based on estimated distance between
            two points on the ground along the azimuth direction (`rdr2geo` method).
            This code may introduce NaNs in the image where rdr2geo fails
            */
            for (int i = 0; i < radar_grid.length(); ++i) {
                for (int j = 0; j < radar_grid.width(); ++j) {
                    /*
                    The target_pos arrays have an extra line at the beginning
                    and an extra line at the end. The variable `ii` indicates
                    the indices for those arrays.  The line `i=0` in the radar
                    grid is located at line `ii=1` in target_pos arrays.
                    */
                    const int ii = i + 1;

                    /*
                    We compute the distance between the azimuth line `ii + 1`
                    and `ii - 1` and divide it by the associated time interval.
                    This distance does not consider the Earth's curvature
                    which should be in the order of 3mm for a 1km distance.
                    */

                    // derivative X wrt az. time
                    double dx_dt = ((target_pos_x(ii + 1, j) -
                                        target_pos_x(ii - 1, j)) /
                                    (2 * radar_grid.azimuthTimeInterval()));

                    // derivative Y wrt az. time
                    double dy_dt = ((target_pos_y(ii + 1, j) -
                                        target_pos_y(ii - 1, j)) /
                                    (2 * radar_grid.azimuthTimeInterval()));

                    // derivative Z wrt az. time
                    double dz_dt = ((target_pos_z(ii + 1, j) -
                                        target_pos_z(ii - 1, j)) /
                                    (2 * radar_grid.azimuthTimeInterval()));

                    ground_track_velocity_array(i, j) = std::sqrt(std::pow(dx_dt, 2) +
                                                                  std::pow(dy_dt, 2) +
                                                                  std::pow(dz_dt, 2));
                }
            }
        }

        writeArray(coordinate_x_raster, coordinate_x_array, height_count);
        writeArray(coordinate_y_raster, coordinate_y_array, height_count);
        writeArray(incidence_angle_raster, incidence_angle_array, height_count);
        writeArray(los_unit_vector_x_raster, los_unit_vector_x_array,
                   height_count);
        writeArray(los_unit_vector_y_raster, los_unit_vector_y_array,
                   height_count);
        writeArray(along_track_unit_vector_x_raster,
                   along_track_unit_vector_x_array, height_count);
        writeArray(along_track_unit_vector_y_raster,
                   along_track_unit_vector_y_array, height_count);
        writeArray(elevation_angle_raster, elevation_angle_array, height_count);
        writeArray(ground_track_velocity_raster, ground_track_velocity_array, 
                   height_count);
    }
}
}
}
