#include <gtest/gtest.h>
#include <isce3/core/Constants.h>
#include <isce3/core/Orbit.h>
#include <isce3/geometry/RTC.h>
#include <isce3/io/IH5.h>
#include <isce3/io/Raster.h>
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/RadarGridParameters.h>
#include <string>

// Create set of RadarGridParameters to process
std::set<std::string> radar_grid_str_set = {"cropped", "multilooked"};

// Create set of rtcAlgorithms
std::set<isce3::geometry::rtcAlgorithm> rtc_algorithm_set = {
        isce3::geometry::rtcAlgorithm::RTC_BILINEAR_DISTRIBUTION,
        isce3::geometry::rtcAlgorithm::RTC_AREA_PROJECTION};

TEST(TestRTC, RunRTC) {
    // Open HDF5 file and load products
    isce3::io::IH5File file(TESTDATA_DIR "envisat.h5");
    isce3::product::RadarGridProduct product(file);
    char frequency = 'A';

    // Open DEM raster
    isce3::io::Raster dem(TESTDATA_DIR "srtm_cropped.tif");

    // Create radar grid parameter
    isce3::product::RadarGridParameters radar_grid_sl(product, frequency);

    // Crop original radar grid parameter
    isce3::product::RadarGridParameters radar_grid_cropped =
            radar_grid_sl.offsetAndResize(30, 135, 128, 128);

    // Multi-look original radar grid parameter
    int nlooks_az = 5, nlooks_rg = 5;
    isce3::product::RadarGridParameters radar_grid_ml =
            radar_grid_sl.multilook(nlooks_az, nlooks_rg);

    // Create orbit and Doppler LUT
    isce3::core::Orbit orbit = product.metadata().orbit();
    isce3::core::LUT2d<double> dop =
            product.metadata().procInfo().dopplerCentroid(frequency);

    dop.boundsError(false);

    // Set input parameters
    isce3::geometry::rtcInputTerrainRadiometry inputTerrainRadiometry =
            isce3::geometry::rtcInputTerrainRadiometry::BETA_NAUGHT;

    isce3::geometry::rtcOutputTerrainRadiometry outputTerrainRadiometry =
            isce3::geometry::rtcOutputTerrainRadiometry::GAMMA_NAUGHT;

    isce3::geometry::rtcAreaMode rtc_area_mode =
            isce3::geometry::rtcAreaMode::AREA_FACTOR;

    isce3::geometry::rtcAreaBetaMode rtc_area_beta_mode =
            isce3::geometry::rtcAreaBetaMode::AUTO;

    for (auto radar_grid_str : radar_grid_str_set) {

        isce3::product::RadarGridParameters radar_grid;

        // Open DEM raster
        if (radar_grid_str == "cropped")
            radar_grid = radar_grid_cropped;
        else
            radar_grid = radar_grid_ml;

        for (auto rtc_algorithm : rtc_algorithm_set) {

            double geogrid_upsampling = 1;

            std::string filename;
            // test removed because it requires high geogrid upsampling (too
            // slow)
            if (rtc_algorithm == isce3::geometry::rtcAlgorithm::
                                         RTC_BILINEAR_DISTRIBUTION &&
                    radar_grid_str == "cropped") {
                continue;
            } else if (rtc_algorithm == isce3::geometry::rtcAlgorithm::
                                                RTC_BILINEAR_DISTRIBUTION) {
                filename = "./rtc_bilinear_distribution_" + radar_grid_str +
                           ".bin";
            } else {
                filename = "./rtc_area_proj_" + radar_grid_str + ".bin";
            }
            std::cout << "generating file: " << filename << std::endl;

            // Create output raster
            isce3::io::Raster out_raster(filename, radar_grid.width(),
                                        radar_grid.length(), 1, GDT_Float32,
                                        "ENVI");

            // Call RTC
            isce3::geometry::computeRtc(radar_grid, orbit, dop, dem, out_raster,
                    inputTerrainRadiometry, outputTerrainRadiometry,
                    rtc_area_mode, rtc_algorithm, rtc_area_beta_mode,
                    geogrid_upsampling);
        }
    }
}

TEST(TestRTC, CheckResults) {

    for (auto radar_grid_str : radar_grid_str_set) {

        for (auto rtc_algorithm : rtc_algorithm_set) {

            double max_rmse;

            std::string filename;

            // test removed because it requires high geogrid upsampling (too
            // slow)
            if (rtc_algorithm == isce3::geometry::rtcAlgorithm::
                                         RTC_BILINEAR_DISTRIBUTION &&
                    radar_grid_str == "cropped") {
                continue;
            } else if (rtc_algorithm == isce3::geometry::rtcAlgorithm::
                                                RTC_BILINEAR_DISTRIBUTION) {
                max_rmse = 0.7;
                filename = "./rtc_bilinear_distribution_" + radar_grid_str +
                           ".bin";
            } else {
                max_rmse = 0.1;
                filename = "./rtc_area_proj_" + radar_grid_str + ".bin";
            }

            std::cout << "evaluating file: " << filename << std::endl;

            // Open computed integrated-area raster
            isce3::io::Raster testRaster(filename);

            // Open reference raster
            std::string ref_filename =
                    TESTDATA_DIR "rtc/rtc_" + radar_grid_str + ".bin";
            isce3::io::Raster refRaster(ref_filename);
            std::cout << "reference file: " << ref_filename << std::endl;

            ASSERT_TRUE(testRaster.width() == refRaster.width() and
                        testRaster.length() == refRaster.length());

            double square_sum = 0; // sum of square difference
            int n_nan = 0;         // number of NaN pixels
            int n_npos = 0;        // number of non-positive pixels

            // Valarray to hold line of data
            std::valarray<double> test(testRaster.width()),
                    ref(refRaster.width());
            int n_valid = 0;
            for (size_t i = 0; i < refRaster.length(); i++) {
                // Get line of data
                testRaster.getLine(test, i, 1);
                refRaster.getLine(ref, i, 1);
                // Check each value in the line
                for (size_t j = 0; j < refRaster.width(); j++) {
                    if (std::isnan(test[j]) or std::isnan(ref[j])) {
                        n_nan++;
                        continue;
                    }
                    if (ref[j] <= 0 or test[j] <= 0) {
                        n_npos++;
                        continue;
                    }
                    n_valid++;
                    square_sum += pow(test[j] - ref[j], 2);
                }
            }
            printf("    ----------------\n");
            printf("    # total: %d\n", n_valid + n_nan + n_npos);
            printf("    ----------------\n");
            printf("    # valid: %d\n", n_valid);
            printf("    # NaNs: %d\n", n_nan);
            printf("    # non-positive: %d\n", n_npos);
            printf("    ----------------\n");

            ASSERT_GT(n_valid, 0);

            // Compute average over entire image
            double rmse = std::sqrt(square_sum / n_valid);

            printf("    RMSE = %g\n", rmse);
            printf("    ----------------\n");
            
            // Enforce bound on average pixel-error
            ASSERT_LT(rmse, max_rmse);

            // Enforce bound on number of ignored pixels
            ASSERT_LT(n_nan, 1e-4 * refRaster.width() * refRaster.length());
            ASSERT_LT(n_npos, 1e-4 * refRaster.width() * refRaster.length());
        }
    }
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
