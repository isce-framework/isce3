#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <isce3/core/LookSide.h>
#include <isce3/geocode/GeocodeCov.h>
#include <isce3/geometry/metadataCubes.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/Topo.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/RadarGridParameters.h>

template<class T>
static isce3::core::Matrix<T> _getCubeArray(isce3::io::Raster& cube_raster,
                                            int length, int width, int band)
{
    isce3::core::Matrix<T> data_array;
    data_array.resize(length, width);
    cube_raster.getBlock(data_array.data(), 0, 0, width, length, band);
    return data_array;
}

void _check_vectors(const isce3::product::GeoGridParameters& geogrid,
                    double height, double band, const isce3::core::Orbit& orbit,
                    isce3::io::Raster& slant_range_raster,
                    isce3::io::Raster& azimuth_time_raster,
                    isce3::io::Raster& incidence_angle_raster,
                    isce3::io::Raster& los_unit_vector_x_raster,
                    isce3::io::Raster& los_unit_vector_y_raster,
                    isce3::io::Raster& along_track_unit_vector_x_raster,
                    isce3::io::Raster& along_track_unit_vector_y_raster,
                    isce3::io::Raster& elevation_angle_raster,
                    isce3::io::Raster& ground_track_velocity_raster)
{
    auto proj = isce3::core::makeProjection(geogrid.epsg());

    const isce3::core::Ellipsoid& ellipsoid = proj->ellipsoid();

    auto slant_range_array = _getCubeArray<double>(
            slant_range_raster, geogrid.length(), geogrid.width(), band);
    auto azimuth_time_array = _getCubeArray<double>(
            azimuth_time_raster, geogrid.length(), geogrid.width(), band);
    auto incidence_angle_array = _getCubeArray<float>(
            incidence_angle_raster, geogrid.length(), geogrid.width(), band);
    auto los_unit_vector_x_array = _getCubeArray<float>(
            los_unit_vector_x_raster, geogrid.length(), geogrid.width(), band);
    auto los_unit_vector_y_array = _getCubeArray<float>(
            los_unit_vector_y_raster, geogrid.length(), geogrid.width(), band);
    auto along_track_unit_vector_x_array =
            _getCubeArray<float>(along_track_unit_vector_x_raster,
                                 geogrid.length(), geogrid.width(), band);
    auto along_track_unit_vector_y_array =
            _getCubeArray<float>(along_track_unit_vector_y_raster,
                                 geogrid.length(), geogrid.width(), band);
    auto elevation_angle_array = _getCubeArray<float>(
            elevation_angle_raster, geogrid.length(), geogrid.width(), band);
    auto ground_track_velocity_array = _getCubeArray<double>(
            ground_track_velocity_raster, geogrid.length(), geogrid.width(), 
            band);

    const double slant_range_error_threshold = 1e-16;
    const double vel_unit_error_threshold = 1e-3;
    const double sat_position_error_threshold = 1e-3;
    const double incidence_angle_error_threshold = 1e-5;
    const double ground_track_velocity_error_threshold = 1e-12;

    for (int i = 0; i < geogrid.length(); ++i) {
        double pos_y = geogrid.startY() + (0.5 + i) * geogrid.spacingY();
        for (int j = 0; j < geogrid.width(); ++j) {
            double pos_x = geogrid.startX() + (0.5 + j) * geogrid.spacingX();

            // Get target coordinates in the output projection system
            const isce3::core::Vec3 target_proj {pos_x, pos_y, height};

            // Get target coordinates in llh
            const isce3::core::Vec3 target_llh = proj->inverse(target_proj);
            const isce3::core::Vec3 target_xyz =
                    ellipsoid.lonLatToXyz(target_llh);

            // Get platform position and velocity from orbit and cube azimuth
            // time
            double azimuth_time = azimuth_time_array(i, j);
            isce3::core::cartesian_t sat_xyz, vel_xyz;
            const isce3::core::Mat3 xyz2enu =
                    isce3::core::Mat3::xyzToEnu(target_llh[1], target_llh[0]);
            const isce3::core::Mat3 enu2xyz =
                    isce3::core::Mat3::enuToXyz(target_llh[1], target_llh[0]);

            isce3::error::ErrorCode status = orbit.interpolate(
                    &sat_xyz, &vel_xyz, azimuth_time,
                    isce3::core::OrbitInterpBorderMode::FillNaN);

            // If interpolation fails, skip
            if (status != isce3::error::ErrorCode::Success) {
                continue;
            }

            /* 1. Evaluate slant-range position */
            double slant_range_ref = (sat_xyz - target_xyz).norm();
            double slant_range_test = slant_range_array(i, j);
            ASSERT_NEAR(slant_range_test, slant_range_ref,
                        slant_range_error_threshold);

            // 2. Compute incidence angle in ENU (geodetic)
            const isce3::core::Vec3 look_vector_xyz =
                    (target_xyz - sat_xyz).normalized();
            const isce3::core::Vec3 look_vector_enu =
                    xyz2enu.dot(look_vector_xyz).normalized();
            const double cos_inc = std::abs(look_vector_enu[2]);
            double incidence_angle_ref = std::acos(cos_inc) * 180.0 / M_PI;
            double incidence_angle_test = incidence_angle_array(i, j);

            /* 2. Evaluate incidence angle in degrees */
            ASSERT_NEAR(incidence_angle_test, incidence_angle_ref,
                        incidence_angle_error_threshold);

            // 3. Estimate platform position from cube LOS vector
            isce3::core::Vec3 los_unit_test, look_vector_ref;
            los_unit_test[0] = los_unit_vector_x_array(i, j);
            los_unit_test[1] = los_unit_vector_y_array(i, j);
            /*
            Obtain height term considering that the vector is unitary and
            the line-of-sight vector points upwards (i.e. height term
            is positive) from the target to the sensor.
            */
            los_unit_test[2] = std::sqrt(
                    std::max(0.0, 1.0 - std::pow(los_unit_test[0], 2) -
                                          std::pow(los_unit_test[1], 2)));

            // 3. Estimate platform position
            isce3::core::Vec3 sat_xyz_test, sat_llh_test, los_xyz_test;
            // Vectors are given in ENU
            los_xyz_test =
                    enu2xyz.dot(los_unit_test).normalized();
            sat_xyz_test = target_xyz + slant_range_test * los_xyz_test;

            /* 
            3. Evaluate platform position from LOS vector and slant-range
            distance extracted from metadata cubes (errors are multiplied).
            */
            ASSERT_NEAR(sat_xyz_test[0], sat_xyz[0],
                        sat_position_error_threshold);
            ASSERT_NEAR(sat_xyz_test[1], sat_xyz[1],
                        sat_position_error_threshold);
            ASSERT_NEAR(sat_xyz_test[2], sat_xyz[2],
                        sat_position_error_threshold);

            // 4. Estimate velocity from cube along-track vector
            isce3::core::Vec3 along_track_unit_vector_test;
            along_track_unit_vector_test[0] =
                    along_track_unit_vector_x_array(i, j);
            along_track_unit_vector_test[1] =
                    along_track_unit_vector_y_array(i, j);

            if (std::isnan(along_track_unit_vector_test[0]) ||
                std::isnan(along_track_unit_vector_test[1])) {
                continue;
            }

            along_track_unit_vector_test[2] = std::sqrt(std::max(
                    0.0, 1.0 - std::pow(along_track_unit_vector_test[0], 2) -
                                 std::pow(along_track_unit_vector_test[1], 2)));

            const isce3::core::Vec3 vel_unit_xyz = vel_xyz.normalized();
            isce3::core::Vec3 along_track_unit_vector_xyz_test;

            /* along-track unit vector is given in ENU */
            along_track_unit_vector_xyz_test =
                    enu2xyz.dot(along_track_unit_vector_test).normalized();

            // 4. Check along-track unit vector
            ASSERT_NEAR(along_track_unit_vector_xyz_test[0], vel_unit_xyz[0],
                        vel_unit_error_threshold);
            ASSERT_NEAR(along_track_unit_vector_xyz_test[1], vel_unit_xyz[1],
                        vel_unit_error_threshold);
            ASSERT_NEAR(along_track_unit_vector_xyz_test[2], vel_unit_xyz[2],
                        vel_unit_error_threshold);

            // 5. Check ground-track velocity vector
            double ground_track_velocity_test = ground_track_velocity_array(i, j);
            const auto target_xyz_normalized = target_xyz.normalized();
            const auto sat_xyz_normalized =  sat_xyz.normalized();
            const double cos_alpha = target_xyz_normalized.dot(sat_xyz_normalized);
            const double ground_track_velocity_ref =
                cos_alpha * target_xyz.norm() * vel_xyz.norm() / sat_xyz.norm();
            ASSERT_NEAR(ground_track_velocity_test, ground_track_velocity_ref,
                        ground_track_velocity_error_threshold);

        }
    }
}

template<class T>
void _compareArrays(isce3::core::Matrix<T>& topo_array,
                    isce3::core::Matrix<T>& cube_array, const int width,
                    const int length, bool flag_geo = false)
{
    double square_error_sum = 0;
    double max_abs_error = 0;
    double sum_topo = 0;
    double sum_cube = 0;

    int nvalid = 0;
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < width; ++j) {
            if (std::isnan(topo_array(i, j)) or std::isnan(cube_array(i, j))) {
                continue;
            }
            const double error = topo_array(i, j) - cube_array(i, j);

            sum_topo += topo_array(i, j);
            sum_cube += cube_array(i, j);
            
            nvalid += 1;
            square_error_sum += std::pow(error, 2);
            max_abs_error = std::max(max_abs_error, std::abs(error));
        }
    }

    std::cout << "mean topo: " << sum_topo / nvalid << std::endl;
    std::cout << "mean cube: " << sum_cube / nvalid << std::endl;

    double percent_valid = nvalid * 100 / (length * width);
    double rmse = std::sqrt(square_error_sum / nvalid);

    std::cout << "     % valid: " << percent_valid << std::endl;
    std::cout << "     RMSE: " << rmse << std::endl;
    std::cout << "     max. abs. error: " << max_abs_error << std::endl;

    if (!flag_geo) {
        ASSERT_TRUE(percent_valid == 100);
    }
    if (flag_geo && std::is_same<T, float>::value) {
        // geo and float
        ASSERT_TRUE(rmse < 1e-5);
        ASSERT_TRUE(max_abs_error < 1e-5);
    } else if (std::is_same<T, float>::value) {
        // slant-range and float
        ASSERT_TRUE(rmse < 1e-7);
        ASSERT_TRUE(max_abs_error < 1e-6);
    } else if (flag_geo) {
        // geo and double
        ASSERT_TRUE(rmse < 1e-7);
        ASSERT_TRUE(max_abs_error < 1e-6);
    } else {
        // slant-range and double
        ASSERT_TRUE(rmse < 1e-15);
        ASSERT_TRUE(max_abs_error < 1e-13);
    }
}

template<class T>
void _compareCubeLayer(isce3::io::Raster& topo_raster, isce3::io::Raster& cube_raster,
                       int layer_counter, bool flag_geo = false)
{
    ASSERT_TRUE(topo_raster.width() == cube_raster.width());
    ASSERT_TRUE(topo_raster.length() == cube_raster.length());

    int length = topo_raster.length();
    int width = topo_raster.width();

    isce3::core::Matrix<T> topo_array(length, width);
    isce3::core::Matrix<T> cube_array(length, width);

    topo_raster.getBlock(topo_array.data(), 0, 0, width, length, 1);
    cube_raster.getBlock(cube_array.data(), 0, 0, width, length,
                         1 + layer_counter);

    _compareArrays(topo_array, cube_array, width, length, flag_geo);
}

template<class T>
void _compareHeading(std::string topo_file, std::string cube_x_file,
                     std::string cube_y_file, int layer_counter,
                     isce3::product::RadarGridParameters& radar_grid,
                     bool flag_along_track_vector = false)
{
    std::cout << "evaluating:" << std::endl;
    std::cout << "    heading file 1: " << cube_x_file
              << " (band: " << layer_counter + 1 << ")" << std::endl;
    std::cout << "    heading file 2: " << cube_y_file
              << " (band: " << layer_counter + 1 << ")" << std::endl;
    std::cout << "    reference: " << topo_file << std::endl;

    isce3::io::Raster topo_raster(topo_file);
    isce3::io::Raster cube_x_raster(cube_x_file);
    isce3::io::Raster cube_y_raster(cube_y_file);

    ASSERT_TRUE(topo_raster.width() == cube_x_raster.width());
    ASSERT_TRUE(topo_raster.length() == cube_x_raster.length());
    ASSERT_TRUE(topo_raster.width() == cube_y_raster.width());
    ASSERT_TRUE(topo_raster.length() == cube_y_raster.length());

    int length = topo_raster.length();
    int width = topo_raster.width();

    isce3::core::Matrix<T> topo_array(length, width);
    isce3::core::Matrix<T> heading_array(length, width);
    isce3::core::Matrix<T> cube_x_array(length, width);
    isce3::core::Matrix<T> cube_y_array(length, width);

    topo_raster.getBlock(topo_array.data(), 0, 0, width, length, 1);
    cube_x_raster.getBlock(cube_x_array.data(), 0, 0, width, length,
                           1 + layer_counter);
    cube_y_raster.getBlock(cube_y_array.data(), 0, 0, width, length,
                           1 + layer_counter);

    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < width; ++j) {
            if (std::isnan(topo_array(i, j)) or
                std::isnan(cube_x_array(i, j)) or
                std::isnan(cube_y_array(i, j))) {
                continue;
            }
            double heading;
            if (flag_along_track_vector) {
                heading = std::atan2(cube_y_array(i, j), cube_x_array(i, j)) *
                          180 / M_PI;
            } else if (radar_grid.lookSide() == isce3::core::LookSide::Right) {
                // Envisat dataset is right looking
                heading = (std::atan2(cube_y_array(i, j), cube_x_array(i, j)) +
                           0.5 * M_PI) *
                          180 / M_PI;
            } else {
                heading = (std::atan2(cube_y_array(i, j), cube_x_array(i, j)) -
                           0.5 * M_PI) *
                          180 / M_PI;
            }

            if (heading > 180) {
                heading -= 360;
            } else if (heading < -180) {
                heading += 360;
            }
            heading_array(i, j) = heading;
        }
    }
    _compareArrays(topo_array, heading_array, width, length);
}

TEST(radarGridCubeTest, testRadarGridCube)
{

    // Open the HDF5 product
    // std::string h5file(TESTDATA_DIR "envisat.h5");
    std::string h5file(TESTDATA_DIR "winnipeg.h5");
    isce3::io::IH5File file(h5file);

    // Load the product
    isce3::product::RadarGridProduct product(file);

    // Create radar grid parameter
    char frequency = 'A';
    isce3::product::RadarGridParameters radar_grid(product, frequency);

    // Create orbit and Doppler LUTs
    isce3::core::Orbit orbit = product.metadata().orbit();

    isce3::core::LUT2d<double> zero_doppler;
    zero_doppler.boundsError(false);

    double threshold_geo2rdr = 1e-8;
    int numiter_geo2rdr = 25;
    double delta_range = 1e-8;

    std::vector<double> heights = {0.0, 1000.0};
    std::vector<int> epsgs = {32614, 4326};

    // Prepare geogrid
    for (auto epsg : epsgs) {
        double y0, x0, dx, dy;
        int width, length;

        if (epsg == 4326) {
            y0 = 49.4829999999999970;
            x0 = -97.7500000000000000;
            dx = 0.0002;
            dy = -0.0002;
            width = 230;
            length = 160;
        } else {
            y0 = 5485000;
            x0 = 587000;
            dx = 100;
            dy = -100;
            width = 100;
            length = 100;
        }

        isce3::io::Raster slant_range_raster("slantRange.rdr", width, length,
                                             heights.size(), GDT_Float64,
                                             "ENVI");
        isce3::io::Raster azimuth_time_raster("zeroDopplerAzimuthTime.rdr",
                                              width, length, heights.size(),
                                              GDT_Float64, "ENVI");
        isce3::io::Raster incidence_angle_raster("incidenceAngle.rdr", width,
                                                 length, heights.size(),
                                                 GDT_Float32, "ENVI");
        isce3::io::Raster los_unit_vector_x_raster("losUnitVectorX.rdr", width,
                                                   length, heights.size(),
                                                   GDT_Float32, "ENVI");
        isce3::io::Raster los_unit_vector_y_raster("losUnitVectorY.rdr", width,
                                                   length, heights.size(),
                                                   GDT_Float32, "ENVI");
        isce3::io::Raster along_track_unit_vector_x_raster(
                "alongTrackUnitVectorX.rdr", width, length, heights.size(),
                GDT_Float32, "ENVI");
        isce3::io::Raster along_track_unit_vector_y_raster(
                "alongTrackUnitVectorY.rdr", width, length, heights.size(),
                GDT_Float32, "ENVI");
        isce3::io::Raster elevation_angle_raster("elevationAngle.rdr", width,
                                                 length, heights.size(),
                                                 GDT_Float32, "ENVI");
        isce3::io::Raster ground_track_velocity_raster(
            "groundTrackVelocity.rdr", width, length, heights.size(),
            GDT_Float64, "ENVI");

        isce3::product::GeoGridParameters geogrid(x0, y0, dx, dy, width, length,
                                                  epsg);

        bool flag_set_output_rasters_geolocation = true;

        // Make cubes
        std::cout << "calling makeRadarGridCubes() with geogrid EPSG:" << epsg
                  << std::endl;
        isce3::geometry::makeRadarGridCubes(radar_grid, geogrid, heights, orbit,
                zero_doppler, zero_doppler,
                &slant_range_raster, &azimuth_time_raster,
                &incidence_angle_raster, &los_unit_vector_x_raster,
                &los_unit_vector_y_raster, &along_track_unit_vector_x_raster,
                &along_track_unit_vector_y_raster, &elevation_angle_raster,
                &ground_track_velocity_raster,
                threshold_geo2rdr, numiter_geo2rdr, delta_range,
                flag_set_output_rasters_geolocation);

        // 1. Check geotransform and EPSG
        std::vector<isce3::io::Raster> raster_vector = {
                slant_range_raster,
                azimuth_time_raster,
                incidence_angle_raster,
                los_unit_vector_x_raster,
                los_unit_vector_y_raster,
                along_track_unit_vector_x_raster,
                along_track_unit_vector_y_raster,
                elevation_angle_raster,
                ground_track_velocity_raster};

        for (auto cube_raster : raster_vector) {
            ASSERT_TRUE(cube_raster.getEPSG() == epsg);
            std::vector<double> cube_geotransform(6);
            cube_raster.getGeoTransform(cube_geotransform);
            std::vector<double> geogrid_geotransform = {geogrid.startX(),
                                                        geogrid.spacingX(),
                                                        0,
                                                        geogrid.startY(),
                                                        0,
                                                        geogrid.spacingY()};
            for (int count = 0; count < geogrid_geotransform.size(); ++count) {
                ASSERT_TRUE(cube_geotransform[count] ==
                            geogrid_geotransform[count]);
            }
        }

        // 2. Reconstruct radar geometry vectors
        std::cout << "checking vectors..." << std::endl;
        for (int height_count = 0; height_count < heights.size();
             ++height_count) {
            _check_vectors(
                    geogrid, heights[height_count], height_count + 1, orbit,
                    slant_range_raster, azimuth_time_raster,
                    incidence_angle_raster, los_unit_vector_x_raster,
                    los_unit_vector_y_raster, along_track_unit_vector_x_raster,
                    along_track_unit_vector_y_raster, elevation_angle_raster,
                    ground_track_velocity_raster);
        }

        // 3. Compare results with topo

        auto proj = isce3::core::makeProjection(epsg);

        isce3::core::Ellipsoid ellipsoid = proj->ellipsoid();

        // Create reference values from topo
        isce3::geometry::Topo topo(radar_grid, orbit, ellipsoid, zero_doppler);

        topo.epsgOut(epsg);

        if (radar_grid.lookSide() == isce3::core::LookSide::Right) {
            std::cout << "look side: Right" << std::endl;
        } else {
            std::cout << "look side: Left" << std::endl;
        }

        std::vector<std::string> files_to_geocode_vector = {
                "tempSlantRange.rdr", "tempZeroDopplerAzimuthTime.rdr",
                "inc.rdr", "hdg.rdr"};
        std::vector<GDALDataType> dtype_vector = {GDT_Float64, GDT_Float64,
                                                  GDT_Float32, GDT_Float32};

        // Geocode object
        isce3::geocode::Geocode<double> geo_obj;

        // The interpolation method used for geocoding
        isce3::core::dataInterpMethod method = isce3::core::BIQUINTIC_METHOD;

        // Manually configure geocode object
        geo_obj.orbit(orbit);
        geo_obj.doppler(zero_doppler);
        geo_obj.ellipsoid(ellipsoid);
        geo_obj.thresholdGeo2rdr(threshold_geo2rdr);
        geo_obj.numiterGeo2rdr(numiter_geo2rdr);
        geo_obj.radarBlockMargin(100);
        geo_obj.dataInterpolator(method);

        geo_obj.geoGrid(x0, y0, dx, dy, width, length, epsg);

        bool flag_geo = true;

        {
            // Create slantRange file for geocoding
            std::cout << "creating temporary slant-range file" << std::endl;
            std::string temp_slant_range_file = "tempSlantRange.rdr";
            isce3::io::Raster temp_slant_range_raster(
                    temp_slant_range_file, radar_grid.width(), radar_grid.length(),
                    1, GDT_Float64, "ENVI");
            isce3::core::Matrix<double> slant_range_array(radar_grid.length(),
                                                          radar_grid.width());
            // Create zeroDopplerAzimuthTime for geocoding
            std::cout << "creating temporary azimuth time file" << std::endl;
            std::string temp_azimuth_time_file = "tempZeroDopplerAzimuthTime.rdr";
            isce3::io::Raster temp_azimuth_time_raster(
                    temp_azimuth_time_file, radar_grid.width(), radar_grid.length(),
                    1, GDT_Float64, "ENVI");
            isce3::core::Matrix<double> azimuth_time_array(radar_grid.length(),
                                                           radar_grid.width());

            for (int i = 0; i < radar_grid.length(); ++i) {
                for (int j = 0; j < radar_grid.width(); ++j) {
                    slant_range_array(i, j) = radar_grid.startingRange() +
                                              j * radar_grid.rangePixelSpacing();
                    azimuth_time_array(i, j) = radar_grid.sensingStart() +
                                               i * radar_grid.azimuthTimeInterval();
                }
            }

            temp_slant_range_raster.setBlock(slant_range_array.data(), 0, 0,
                                             radar_grid.width(),
                                             radar_grid.length(), 1);

            temp_azimuth_time_raster.setBlock(azimuth_time_array.data(), 0, 0,
                                              radar_grid.length(),
                                              radar_grid.width(), 1);
        }

        // DEM for UAVSAR (NISAR) Winipeg
        int epsg_dem = epsg;
        double dem_x0, dem_y0, dem_dx, dem_dy;
        int dem_length, dem_width;
        if (epsg_dem == 4326) {
            dem_x0 = -98.44;
            dem_y0 = 49.995;
            dem_dx = 0.002;
            dem_dy = -0.002;
            dem_length = 400;
            dem_width = 400;
        } else {
            dem_x0 = 586200;
            dem_y0 = 5486000;
            dem_dx = 100;
            dem_dy = -100;
            dem_width = 120;
            dem_length = 120;
        }

        for (std::size_t layer_counter = 0; layer_counter < heights.size();
             ++layer_counter) {

            auto height = heights[layer_counter];
            std::cout << "preparing DEM interpolator for height: " << height
                      << std::endl;
            isce3::geometry::DEMInterpolator dem(height);

            // Setup DEM interpolator
            dem.epsgCode(epsg_dem);
            dem.xStart(dem_x0);
            dem.yStart(dem_y0);
            dem.deltaX(dem_dx);
            dem.deltaY(dem_dy);
            dem.length(dem_length);
            dem.width(dem_width);

            // Run topo
            const std::string outdir = ".";
            topo.topo(dem, outdir);

            // Create DEM for geocoding
            std::string temp_dem_file = "temp_dem.bin";
            isce3::io::Raster dem_raster(temp_dem_file, dem_width, dem_length,
                                         1, GDT_Float32, "ENVI");
            isce3::core::Matrix<float> dem_array(dem_length, dem_width);
            dem_array.fill(height);
            dem_raster.setBlock(dem_array.data(), 0, 0, dem_width, dem_length,
                                1);
            double geotransform[] = {dem_x0, dem_dx, 0, dem_y0, 0, dem_dy};
            dem_raster.setGeoTransform(geotransform);
            dem_raster.setEPSG(epsg);

            for (int count = 0; count < files_to_geocode_vector.size();
                 ++count) {

                std::string file_to_geocode = files_to_geocode_vector[count];

                std::string output_geocoded_file = file_to_geocode;

                auto start_pos = output_geocoded_file.find(".rdr");
                output_geocoded_file.replace(start_pos, 4, "Geo.bin");

                GDALDataType dtype = dtype_vector[count];

                std::cout << "geocoding file: " << file_to_geocode << std::endl;
                std::cout << "output geocoded file: " << output_geocoded_file
                          << std::endl;

                isce3::geocode::geocodeOutputMode output_mode =
                        isce3::geocode::geocodeOutputMode::INTERP;

                isce3::io::Raster file_to_geocode_raster(file_to_geocode);

                // Create output geocoded raster
                isce3::io::Raster output_geocoded_raster(
                        output_geocoded_file, width, length, 1, dtype, "ENVI");

                // Run geocode
                geo_obj.geocode(radar_grid, file_to_geocode_raster,
                                output_geocoded_raster, dem_raster,
                                output_mode);
            }


            auto inc_angle_ref_raster = isce3::io::Raster("incGeo.bin");
            _compareCubeLayer<float>(inc_angle_ref_raster, incidence_angle_raster,
                                     layer_counter, flag_geo);

            auto slant_range_ref_raster = isce3::io::Raster("tempSlantRangeGeo.bin");
            _compareCubeLayer<double>(slant_range_ref_raster, slant_range_raster,
                                      layer_counter, flag_geo);

            auto azimuth_time_ref_raster =
                    isce3::io::Raster("tempZeroDopplerAzimuthTimeGeo.bin");
            _compareCubeLayer<double>(azimuth_time_ref_raster,
                                      azimuth_time_raster,
                                      layer_counter, flag_geo);
        }
    }
}


TEST(metadataCubesTest, testMetadataCubes) {

    // Open the HDF5 product
    // std::string h5file(TESTDATA_DIR "envisat.h5");
    std::string h5file(TESTDATA_DIR "winnipeg.h5");
    isce3::io::IH5File file(h5file);

    // Load the product
    isce3::product::RadarGridProduct product(file);

    // Create radar grid parameter
    char frequency = 'A';
    isce3::product::RadarGridParameters radar_grid(product, frequency);

    // Create orbit and Doppler LUTs
    isce3::core::Orbit orbit = product.metadata().orbit();

    isce3::core::LUT2d<double> zero_doppler;

    double threshold_geo2rdr = 1e-8;
    int numiter_geo2rdr = 25;
    double delta_range = 1e-6;
    int epsg = 4326;

    std::vector<double> heights = {0.0, 1000.0};

    int length = radar_grid.length();
    int width = radar_grid.width();

    isce3::io::Raster coordinate_x_raster("coordinateX.bin", width, length,
                                          heights.size(), GDT_Float64, "ENVI");
    isce3::io::Raster coordinate_y_raster("coordinateY.bin", width, length,
                                          heights.size(), GDT_Float64, "ENVI");
    isce3::io::Raster incidence_angle_raster("incidenceAngle.bin", width,
                                             length, heights.size(),
                                             GDT_Float32, "ENVI");
    isce3::io::Raster los_unit_vector_x_raster("losUnitVectorX.bin", width,
                                               length, heights.size(),
                                               GDT_Float32, "ENVI");
    isce3::io::Raster los_unit_vector_y_raster("losUnitVectorY.bin", width,
                                               length, heights.size(),
                                               GDT_Float32, "ENVI");
    isce3::io::Raster along_track_unit_vector_x_raster(
            "alongTrackUnitVectorX.bin", width, length, heights.size(),
            GDT_Float32, "ENVI");
    isce3::io::Raster along_track_unit_vector_y_raster(
            "alongTrackUnitVectorY.bin", width, length, heights.size(),
            GDT_Float32, "ENVI");
    isce3::io::Raster elevation_angle_raster("elevationAngle.bin", width,
                                             length, heights.size(),
                                             GDT_Float32, "ENVI");
    isce3::io::Raster ground_track_velocity_raster(
        "groundTrackVelocity.bin", width, length, heights.size(),
        GDT_Float32, "ENVI");

    // Make cubes
    isce3::geometry::makeGeolocationGridCubes(radar_grid, heights, orbit,
            zero_doppler, zero_doppler, epsg,
            &coordinate_x_raster, &coordinate_y_raster, &incidence_angle_raster,
            &los_unit_vector_x_raster, &los_unit_vector_y_raster,
            &along_track_unit_vector_x_raster,
            &along_track_unit_vector_y_raster, &elevation_angle_raster,
            &ground_track_velocity_raster, threshold_geo2rdr, numiter_geo2rdr,
            delta_range);

    auto proj = isce3::core::makeProjection(epsg);

    const isce3::core::Ellipsoid &ellipsoid = proj->ellipsoid();
    
    // create reference values from topo
    isce3::geometry::Topo topo(radar_grid,
                              orbit,
                              ellipsoid,
                              zero_doppler);

    topo.epsgOut(4326);

    if (radar_grid.lookSide() == isce3::core::LookSide::Right) {
        std::cout << "look side: Right" << std::endl;
    } else {
        std::cout << "look side: Left" << std::endl;
    }

    for (std::size_t layer_counter = 0; layer_counter < heights.size();
         ++layer_counter) {
        std::cout << "preparing DEM interpolator for height: "
                  << heights[layer_counter] << std::endl;
        isce3::geometry::DEMInterpolator dem(heights[layer_counter]);

        // UAVSAR (NISAR) Winipeg
        dem.epsgCode(4326);
        dem.xStart(-98.444);
        dem.yStart(49.991);
        dem.deltaX(0.0002);
        dem.deltaY(-0.0002);
        dem.length(3240);
        dem.width(3805);

        const std::string outdir = ".";
        std::cout << "running topo for height: " << heights[layer_counter]
                  << std::endl;
        topo.topo(dem, outdir);
        std::cout << "... done running topo for height: "
                  << heights[layer_counter] << std::endl;

        auto inc_angle_ref_raster = isce3::io::Raster("inc.rdr");
        _compareCubeLayer<float>(inc_angle_ref_raster, incidence_angle_raster,
                                  layer_counter);

        auto x_ref_raster = isce3::io::Raster("x.rdr");
        _compareCubeLayer<double>(x_ref_raster, coordinate_x_raster,
                                  layer_counter);

        auto y_ref_raster = isce3::io::Raster("y.rdr");
        _compareCubeLayer<double>(y_ref_raster, coordinate_y_raster,
                                  layer_counter);

        /*
        Uncomment these lines after the estimation of the heading angle
        from topo is updated. Topo currently derives the heading angle 
        from the look vector instead of deriving it from the velocity
        vector. This result in a low accuracy that cannot be used to
        evaluate the metadata cubes.

         bool flag_along_track_vector = true;
        _compareHeading<float>("hdg.rdr", "alongTrackUnitVectorX.bin",
                               "alongTrackUnitVectorY.bin", layer_counter,
                               radar_grid,
                               flag_along_track_vector);
        
        _compareHeading<float>("hdg.rdr", "losUnitVectorX.bin",
                               "losUnitVectorY.bin", layer_counter,
                               radar_grid);
        */
        
                               
    }
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
