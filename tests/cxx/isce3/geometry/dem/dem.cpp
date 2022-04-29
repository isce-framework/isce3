//-*- C++ -*-

#include <iostream>
#include <string>
#include <cmath>
#include <gtest/gtest.h>

// isce3::core
#include <isce3/core/Constants.h>

// isce3::io
#include <isce3/io/Raster.h>

// isce3::geometry
#include <isce3/geometry/DEMInterpolator.h>


TEST(DEMTest, ConstDEM) {

    //Create a constant height DEM
    float consthgt = 150.0;
    isce3::geometry::DEMInterpolator dem(consthgt);

    //Check for initialization
    EXPECT_NEAR(dem.refHeight(), consthgt, 1.0e-6);
}

TEST(DEMTest, MethodConstruct) {

    //Constant height
    float consthgt = 220.0;

    //Methods to iterate over
    std::vector<isce3::core::dataInterpMethod> methods = { isce3::core::SINC_METHOD,
                                                          isce3::core::BILINEAR_METHOD,
                                                          isce3::core::BICUBIC_METHOD,
                                                          isce3::core::NEAREST_METHOD,
                                                          isce3::core::BIQUINTIC_METHOD };

    for(auto &method: methods)
    {
        isce3::geometry::DEMInterpolator dem(consthgt, method);
        EXPECT_NEAR(dem.refHeight(), consthgt, 1.0e-6);
        EXPECT_EQ(dem.interpMethod(), method);
    }
}


void test_dateline(double x0, double xf, double y0, double yf,
                   double sampling_factor) {
    /*
    This function uses two geoid rasters (EGM96) to test the DEM
    interpolator for dateline crossing. The geoid raster "egm96_15.gtx"
    is geolocated over geographic coordinates with longitude range 
    varying from -180 to 180 deg whereas the geoid raster
    "egm96_15_lon_0_360.gtx" is the "shifted" version of "egm96_15.gtx"
    geolocated over geographic coordinates with longitude from 0 to 360 deg.
    We compare the interpolated values from DEMInterpolator methods
    interpolateLonLat() and interpolateXY().
    The `sampling_factor` determines the lat/lon step w.r.t. the geoid
    rasters pixel size.
    We compare the values from the four sources:
    1 - Non-interpolated "egm96_15.gtx" values;
    2 - Non-interpolated "egm96_15_lon_0_360.gtx" values;
    3 - DEMInterpolator "egm96_15.gtx" values;
    4 - DEMInterpolator "egm96_15_lon_0_360.gtx" values.
    The non-interpolated values 1 and 2 are only compared to the
    DEMInterpolator values 3 and 4 for the points located in the
    center of the pixel (where no interpolation is required). 
    */
    
    std::cout << "testing DEM interpolator with bbox:" << std::endl;
    std::cout << "    start lon: " << x0 << std::endl;
    std::cout << "    end lon: " << xf << std::endl;
    std::cout << "    start lat: " << y0 << std::endl;
    std::cout << "    end lat: " << yf << std::endl;

    // create geoid raster objects
    isce3::io::Raster raster_geoid(TESTDATA_DIR "egm96_15.gtx");
    isce3::io::Raster raster_geoid_0_to_360(TESTDATA_DIR "egm96_15_lon_0_360.gtx");

    // setup geoid DEM interpolators
    isce3::geometry::DEMInterpolator dem_interp_geoid(0,
                               isce3::core::dataInterpMethod::BIQUINTIC_METHOD);
    auto ret_1 = dem_interp_geoid.loadDEM(raster_geoid, x0, xf, y0, yf);
    if (ret_1 != isce3::error::ErrorCode::Success) {
        throw std::runtime_error("loadDEM failed");
    }

    isce3::geometry::DEMInterpolator dem_interp_geoid_0_to_360(0,
                               isce3::core::dataInterpMethod::BIQUINTIC_METHOD);
    auto ret_2 = dem_interp_geoid_0_to_360.loadDEM(raster_geoid_0_to_360, x0, xf, y0, 
                                      yf);
    if (ret_2 != isce3::error::ErrorCode::Success) {
        throw std::runtime_error("loadDEM failed");
    }

    // read geoid raster
    auto width = raster_geoid.width();
    auto length = raster_geoid.length();
    auto dx = raster_geoid.dx();
    auto dy = raster_geoid.dy();
    ASSERT_GT(dx, 0);
    ASSERT_LT(dy, 0);

    isce3::core::Matrix<float> geoid_array(length, width);
    raster_geoid.getBlock(geoid_array.data(), 0, 0, width, length, 1);

    isce3::core::Matrix<float> geoid_array_0_to_360(length, width);
    raster_geoid_0_to_360.getBlock(geoid_array_0_to_360.data(), 0, 0,
        width, length, 1);
    
    const int interpolation_margin = 5;
    const double err_tolerance = 1e-6;

    // Fix `xf` to be used by for loop
    if (xf < x0) {
        xf += 360;
    }

    for (double lat = yf + interpolation_margin * dy;
         lat > y0 - interpolation_margin * dy;
         lat += sampling_factor * dy) {

        int lat_idx = (lat - 90) / dy;

        for (double lon = x0 + interpolation_margin * dx;
             lon < xf - interpolation_margin * dx;
             lon += sampling_factor * dx) {

            /*
            The non-interpolated values `geoid_array` and
            `geoid_array_0_to_360` are only compared to the DEMInterpolator
            values 3 and 4 for the points located in the center of the pixel
            (where no interpolation is required). This condition is
            represented by the flag `flag_check_arrays`.
            */
            bool flag_check_arrays = 
                (std::fmod(lat, dy) == 0.0) && 
                (std::fmod(lon, dx) == 0.0) &&
                lat_idx < length;

            int lon_idx;
            if (flag_check_arrays) {

                // Wrap `lon` to longitude range [-180, 360]
                double lon_wrapped = lon;
                if (lon > 360 || lon < -360) {
                    lon_wrapped = std::fmod(lon, 360);
                }

                if (lon < -180 - dx) {
                    lon_wrapped += 360;
                }

                if (lon_wrapped < 180) {
                    lon_idx = (lon_wrapped + 180) / dx;
                } else {
                    lon_idx = (lon_wrapped - 180) / dx; 
                }

                int lon_idx_0_to_360;
                if (lon_wrapped >= 0) {
                    lon_idx_0_to_360 = lon_wrapped / dx;
                } else {
                    lon_idx_0_to_360 = (lon_wrapped + 360) / dx;
                }

                /* Check if indexes `lon_idx` and `lon_idx_0_to_360`
                are within arrays' dimensions. If not, set
                `flag_check_arrays` to false.
                */
                if (lon_idx >= width || lon_idx_0_to_360 >= width) {
                    flag_check_arrays = false;
                } else {
                    // compare raster values (without interpolation)
                    ASSERT_NEAR(geoid_array(lat_idx, lon_idx),
                       geoid_array_0_to_360(lat_idx, lon_idx_0_to_360),
                       err_tolerance);
                }
            }

            // test interpolateLonLat()
            double deg_to_rad_factor = M_PI / 180.0;
            double geoid_value = dem_interp_geoid.interpolateLonLat(
                lon * deg_to_rad_factor, lat * deg_to_rad_factor);

            double geoid_0_to_360_value =
                dem_interp_geoid_0_to_360.interpolateLonLat(
                    lon * deg_to_rad_factor, lat * deg_to_rad_factor);

            if (flag_check_arrays) {
                ASSERT_NEAR(geoid_array(lat_idx, lon_idx),
                            geoid_value, err_tolerance);
                ASSERT_NEAR(geoid_array(
                    lat_idx, lon_idx), geoid_0_to_360_value,
                    err_tolerance);
            }

            // test interpolateXY()
            double geoid_value_xy = dem_interp_geoid.interpolateXY(lon, lat);
            double geoid_0_to_360_value_xy =
                dem_interp_geoid_0_to_360.interpolateXY(lon, lat);

            ASSERT_NEAR(geoid_value, geoid_0_to_360_value, err_tolerance);
            ASSERT_NEAR(
                geoid_value_xy, geoid_0_to_360_value_xy, err_tolerance);
        }
    }
}


TEST(DEMTest, DatelineCrossing) {

    std::cout << "dateline crossing test" << std::endl;

    double y0 = -90, yf = 90;

    // offset test wrapping of longitude coordinates around 360 degrees
    for (int offset = -360; offset <= 360; offset += 360) {
        std::cout << "offset:" << offset << std::endl;

        // global with longitude range [-180, 180]
        double sampling_factor = 4;
        double x0 = -180 + offset;
        double xf = 180 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        // global with longitude range [0, 360]
        x0 = 0 + offset;
        xf = 360 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        // longitude values within ]-180, 0[
        x0 = -170 + offset;
        xf = -10 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        // longitude values within ]0, 180[
        x0 = 10 + offset;
        xf = 170 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        // longitude values within ]180, 360[
        x0 = 190 + offset;
        xf = 350 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        // dateline 179.5 to 180.5
        sampling_factor = 0.05;
        x0 = 179.5 + offset;
        xf = 180.5 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        // test DEM right edges
        x0 = 179 + offset;
        xf = 179.875 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        x0 = 359 + offset;
        xf = 359.875 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        // test DEM right edges (global)
        sampling_factor = 8;
        x0 = -180 + offset;
        xf = 179.875 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        x0 = 0 + offset;
        xf = 359.875 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        // test DEM left edges
        sampling_factor = 0.05;
        x0 = -180.125 + offset;
        xf = -179 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        x0 = -0.125 + offset;
        xf = 1.0 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        // test DEM left edges (global)
        sampling_factor = 8;
        x0 = -180.125 + offset;
        xf = 179.875 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        x0 = -0.125 + offset;
        xf = 359.875 + offset;
        test_dateline(x0, xf, y0, yf, sampling_factor);

        if (offset == 0) {

            // dateline 179.5 to -179.5
            x0 = 179.5 + offset;
            xf = -179.5 + offset;
            test_dateline(x0, xf, y0, yf, sampling_factor);

        }
    }
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
