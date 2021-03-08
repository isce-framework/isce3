#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Metadata.h>
#include <isce3/core/Orbit.h>
#include <isce3/geocode/geocodeSlc.h>
#include <isce3/geocode/GeocodeCov.h>
#include <isce3/geometry/Topo.h>
#include <isce3/io/IH5.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/product/Product.h>
#include <isce3/product/Serialization.h>

std::set<std::string> geocode_mode_set = {"interp", "area_proj"};

// Declaration for utility function to read metadata stream from V  RT
std::stringstream streamFromVRT(const char* filename, int bandNum = 1);

// To create a zero height DEM
void createZeroDem();

// To create test data
void createTestData();


TEST(GeocodeTest, TestGeocodeCov) {

    // This test runs Topo to compute lat lon height on ellipsoid for a given
    // radar dataset. Then each of the computed latitude and longitude
    // grids (radar grids) get geocoded. This will allow to check geocoding by
    // comparing the values of the geocoded pixels with its coordinate.

    // Create a DEM with zero height (ellipsoid surface)
    createZeroDem();

    // Run Topo with the zero height DEM and cerate the lat-lon grids on ellipsoid
    createTestData();

    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Load the product
    isce3::product::Product product(file);

    const isce3::product::Swath & swath = product.swath('A');
    isce3::core::Orbit orbit = product.metadata().orbit();
    isce3::core::Ellipsoid ellipsoid;
    isce3::core::LUT2d<double> doppler = product.metadata().procInfo().dopplerCentroid('A');
    auto lookSide = product.lookSide();

    double threshold = 1.0e-9 ;
    int numiter = 25;
    size_t linesPerBlock = 1000;
    double demBlockMargin = 0.1;
    int radarBlockMargin = 10;

    // output geocoded grid (can be different from DEM)
    double geoGridStartX = -115.6;
    double geoGridStartY = 34.832;

    int reduction_factor = 10;

    double geoGridSpacingX = reduction_factor * 0.0002;
    double geoGridSpacingY = reduction_factor * -8.0e-5;
    int geoGridLength = 380 / reduction_factor;
    int geoGridWidth = 400 / reduction_factor;
    int epsgcode = 4326;

    // The DEM to be used for geocoding
    isce3::io::Raster demRaster("zero_height_dem_geo.bin");

    // The interpolation method used for geocoding
    isce3::core::dataInterpMethod method = isce3::core::BIQUINTIC_METHOD;

    // Geocode object
    isce3::geocode::Geocode<double> geoObj;

    // manually configure geoObj

    geoObj.orbit(orbit);
    geoObj.doppler(doppler);
    geoObj.ellipsoid(ellipsoid);
    geoObj.thresholdGeo2rdr(threshold);
    geoObj.numiterGeo2rdr(numiter);
    geoObj.linesPerBlock(linesPerBlock);
    geoObj.demBlockMargin(demBlockMargin);
    geoObj.radarBlockMargin(radarBlockMargin);
    geoObj.interpolator(method);

    isce3::product::RadarGridParameters radar_grid(swath, lookSide);

    geoObj.geoGrid(geoGridStartX, geoGridStartY, geoGridSpacingX,
                   geoGridSpacingY, geoGridWidth, geoGridLength, epsgcode);

    // populate optional parameters
    double geogrid_upsampling = 1;
    bool flag_upsample_radar_grid = false;
    isce3::geometry::rtcInputTerrainRadiometry input_terrain_radiometry =
            isce3::geometry::rtcInputTerrainRadiometry::BETA_NAUGHT;
    isce3::geometry::rtcOutputTerrainRadiometry output_terrain_radiometry =
        isce3::geometry::rtcOutputTerrainRadiometry::GAMMA_NAUGHT;
    int exponent = 0;
    float rtc_min_value_db = std::numeric_limits<float>::quiet_NaN();
    double rtc_geogrid_upsampling =
            std::numeric_limits<double>::quiet_NaN();
    isce3::geometry::rtcAlgorithm rtc_algorithm =
            isce3::geometry::rtcAlgorithm::RTC_AREA_PROJECTION;
    double abs_cal_factor = 1;
    float clip_min = std::numeric_limits<float>::quiet_NaN();
    float clip_max = std::numeric_limits<float>::quiet_NaN();
    float min_nlooks = std::numeric_limits<float>::quiet_NaN();
    float radar_grid_nlooks = 1;
    bool flag_apply_rtc = false;

    isce3::io::Raster* out_geo_rdr = nullptr;
    isce3::io::Raster* out_geo_dem = nullptr;
    isce3::io::Raster* out_geo_nlooks = nullptr;
    isce3::io::Raster* out_geo_rtc = nullptr;
    isce3::io::Raster* input_rtc = nullptr;
    isce3::io::Raster* output_rtc = nullptr;
    isce3::geocode::geocodeMemoryMode geocode_memory_mode_1 =
            isce3::geocode::geocodeMemoryMode::BLOCKS_GEOGRID;
    isce3::geocode::geocodeMemoryMode geocode_memory_mode_2 =
            isce3::geocode::geocodeMemoryMode::BLOCKS_GEOGRID_AND_RADARGRID;

    isce3::geocode::geocodeOutputMode output_mode;

    // test small block size
    const int min_block_size = 16;
    const int max_block_size = isce3::geometry::AP_DEFAULT_MIN_BLOCK_SIZE;

    for (auto geocode_mode_str : geocode_mode_set) {

        std::cout << "geocode_mode: " << geocode_mode_str << std::endl;

        if (geocode_mode_str == "interp")
            output_mode = isce3::geocode::geocodeOutputMode::INTERP;
        else
            output_mode = isce3::geocode::geocodeOutputMode::AREA_PROJECTION;

        for (std::string xy_str : {"x", "y"}) {

            // input raster in radar coordinates to be geocoded
            isce3::io::Raster radarRaster(xy_str + ".rdr");
            std::cout << "geocoding file: " << xy_str + ".rdr" << std::endl;

            // output raster
            isce3::io::Raster geocodedRaster(
                    xy_str + "_" + geocode_mode_str + "_geo.bin", geoGridWidth,
                    geoGridLength, 1, GDT_Float64, "ENVI");

            // run geocode
            geoObj.geocode(radar_grid, radarRaster, geocodedRaster, demRaster,
                           output_mode, geogrid_upsampling,
                           flag_upsample_radar_grid, flag_apply_rtc,
                           input_terrain_radiometry,
                           output_terrain_radiometry, exponent,
                           rtc_min_value_db, rtc_geogrid_upsampling,
                           rtc_algorithm, abs_cal_factor, clip_min, clip_max,
                           min_nlooks, radar_grid_nlooks,
                           nullptr, out_geo_rdr,
                           out_geo_dem, out_geo_nlooks, out_geo_rtc,
                           input_rtc, output_rtc, geocode_memory_mode_1,
                           min_block_size, max_block_size);
        }
    } 

    // Test generation of full-covariance elements and block processing

    // Geocode object
    isce3::geocode::Geocode<std::complex<float>> geoComplexObj;

    // manually configure geoComplexObj
    geoComplexObj.orbit(orbit);
    geoComplexObj.doppler(doppler);
    geoComplexObj.ellipsoid(ellipsoid);
    geoComplexObj.thresholdGeo2rdr(threshold);
    geoComplexObj.numiterGeo2rdr(numiter);
    geoComplexObj.linesPerBlock(linesPerBlock);
    geoComplexObj.demBlockMargin(demBlockMargin);
    geoComplexObj.radarBlockMargin(radarBlockMargin);
    geoComplexObj.interpolator(method);

    geoComplexObj.geoGrid(geoGridStartX, geoGridStartY, geoGridSpacingX,
                          geoGridSpacingY, geoGridWidth, geoGridLength,
                          epsgcode);

    // load complex raster X and Y as a raster vector
    std::vector<Raster> slc_raster_xyVect = {isce3::io::Raster("xslc_rdr.bin"),
                                           isce3::io::Raster("yslc_rdr.bin")};

    isce3::io::Raster slc_raster_xy =
            isce3::io::Raster("xy_slc_rdr.vrt", slc_raster_xyVect);

    // geocode full-covariance
    output_mode = isce3::geocode::geocodeOutputMode::AREA_PROJECTION;

    isce3::io::Raster geocoded_diag_raster("area_proj_geo_diag.bin", geoGridWidth,
                                         geoGridLength, 2, GDT_Float32, "ENVI");

    isce3::io::Raster geocoded_off_diag_raster("area_proj_geo_off_diag.bin",
                                            geoGridWidth, geoGridLength, 1,
                                            GDT_CFloat32, "ENVI");

    geoComplexObj.geocode(
            radar_grid, slc_raster_xy, geocoded_diag_raster, demRaster,
            output_mode, geogrid_upsampling, flag_upsample_radar_grid,
            flag_apply_rtc,
            input_terrain_radiometry, output_terrain_radiometry, 
            exponent, rtc_min_value_db, rtc_geogrid_upsampling,
            rtc_algorithm, abs_cal_factor, clip_min,
            clip_max, min_nlooks, radar_grid_nlooks, &geocoded_off_diag_raster,
            out_geo_rdr, out_geo_dem, out_geo_nlooks, out_geo_rtc,
            input_rtc, output_rtc, geocode_memory_mode_2, min_block_size,
            max_block_size);

    //  load complex raster containing X conj(Y)
    isce3::io::Raster slc_x_conj_y_raster("x_conj_y_slc_rdr.bin");

    isce3::io::Raster geocoded_slc_x_conj_y_raster("area_proj_geo_x_conj_y.bin",
                                                   geoGridWidth, geoGridLength,
                                                   1, GDT_CFloat32, "ENVI");

    geoComplexObj.geocode(radar_grid, slc_x_conj_y_raster, geocoded_slc_x_conj_y_raster,
                          demRaster, output_mode);
}



TEST(GeocodeTest, CheckGeocodeCovFullCovResults) {
    // The geocoded latitude and longitude data should be
    // consistent with the geocoded pixel location.

    std::cout << "opening: area_proj_geo_diag.bin" << std::endl;
    isce3::io::Raster geocoded_diag_raster("area_proj_geo_diag.bin");
    std::cout << "opening: area_proj_geo_off_diag.bin" << std::endl;
    isce3::io::Raster geocoded_off_diag_raster("area_proj_geo_off_diag.bin");
    std::cout << "opening: area_proj_geo_x_conj_y.bin" << std::endl;
    isce3::io::Raster geocoded_slc_x_conj_y_raster("area_proj_geo_x_conj_y.bin");

    size_t length = geocoded_diag_raster.length();
    size_t width = geocoded_diag_raster.width();

    float err_x = 0.0;
    float err_y = 0.0;
    float err_x_conj_y = 0.0;
    float max_err_x = 0.0;
    float max_err_y = 0.0; 
    float max_err_x_conj_y = 0.0;

    std::valarray<float> geocoded_diag_array_x(length * width);
    std::valarray<float> geocoded_diag_array_y(length * width);
    std::valarray<std::complex<float>> geocoded_off_diag_array(length * width);
    std::valarray<std::complex<float>> slc_x_conj_y_array(length * width);

    geocoded_diag_raster.getBlock(geocoded_diag_array_x, 0, 0, width, length, 1);
    geocoded_diag_raster.getBlock(geocoded_diag_array_y, 0, 0, width, length, 2);
    geocoded_off_diag_raster.getBlock(geocoded_off_diag_array, 0, 0, width,
                                    length);
    geocoded_slc_x_conj_y_raster.getBlock(slc_x_conj_y_array, 0, 0, width, length);

    double square_sum_x = 0; // sum of square differences
    int nvalid_x = 0;
    double square_sum_y = 0; // sum of square differences
    int nvalid_y = 0;
    double square_sum_x_conj_y = 0; // sum of square differences
    int nvalid_x_conj_y = 0;

    for (size_t line = 0; line < length; ++line) {
        for (size_t pixel = 0; pixel < width; ++pixel) {
            size_t index = line * width + pixel;

            // < exp(j k x) > = 1
            if (!isnan(geocoded_diag_array_x[index])) {
                err_x = geocoded_diag_array_x[index] - 1;
                square_sum_x += pow(err_x, 2);
                if (geocoded_diag_array_x[index] > 0) {
                    nvalid_x++;
                }
                if (std::abs(err_x) > max_err_x) {
                    max_err_x = std::abs(err_x);
                }
            }

            // < exp(j k y) > = 1
            if (!isnan(geocoded_diag_array_y[index])) {
                err_y = geocoded_diag_array_y[index] - 1;
                square_sum_y += pow(err_y, 2);
                if (geocoded_diag_array_y[index] > 0) {
                    nvalid_y++;
                }
                if (std::abs(err_y) > max_err_y) {
                    max_err_y = std::abs(err_y);
                }
            }

            // geocoded off-diag ~= geocoded x conj (y)
            float norm_off_diag = std::norm(geocoded_off_diag_array[index]);
            float norm_x_conj_y = std::norm(slc_x_conj_y_array[index]);
            if (!isnan(norm_off_diag) && !isnan(norm_x_conj_y)) {
                err_x_conj_y = std::norm(geocoded_off_diag_array[index] -
                                         slc_x_conj_y_array[index]);
                square_sum_x_conj_y += pow(err_x_conj_y, 2);
                if (norm_off_diag > 0 && norm_x_conj_y > 0) {
                    nvalid_x_conj_y++;
                }
                if (err_x_conj_y > max_err_x_conj_y) {
                    max_err_x_conj_y = std::abs(err_x_conj_y);
                }
            }
        }
    }

    double rmse_x = std::sqrt(square_sum_x / nvalid_x);
    double rmse_y = std::sqrt(square_sum_y / nvalid_y);
    double rmse_x_conj_y = std::sqrt(square_sum_x_conj_y / nvalid_x_conj_y);

    std::cout << "geocode_ full-covariance results: " << std::endl;
    std::cout << "  nvalid X: " << nvalid_x << std::endl;
    std::cout << "  nvalid Y: " << nvalid_y << std::endl;
    std::cout << "  nvalid X conj(Y): " << nvalid_x_conj_y << std::endl;
    std::cout << "  RMSE X: " << rmse_x << std::endl;
    std::cout << "  RMSE Y: " << rmse_y << std::endl;
    std::cout << "  RMSE X conj(Y): " << rmse_x_conj_y << std::endl;
    std::cout << "  max err X: " << max_err_x << std::endl;
    std::cout << "  max err Y: " << max_err_y << std::endl;
    std::cout << "  max err X conj(Y): " << max_err_x_conj_y << std::endl;

    ASSERT_GE(nvalid_x, 800);
    ASSERT_GE(nvalid_y, 800);
    ASSERT_GE(nvalid_x_conj_y, 800);

    ASSERT_LT(max_err_x, 1.0e-6);
    ASSERT_LT(max_err_y, 1.0e-6);
    ASSERT_LT(max_err_x_conj_y, 1.0e-6);

}


TEST(GeocodeTest, CheckGeocodeCovResults) {
    // The geocoded latitude and longitude data should be
    // consistent with the geocoded pixel location.

    for (auto geocode_mode_str : geocode_mode_set) {

        isce3::io::Raster xRaster("x_" + geocode_mode_str + "_geo.bin");
        isce3::io::Raster yRaster("y_" + geocode_mode_str + "_geo.bin");
        size_t length = xRaster.length();
        size_t width = xRaster.width();

        double geoTrans[6];
        xRaster.getGeoTransform(geoTrans);

        double x0 = geoTrans[0] + geoTrans[1] / 2.0;
        double dx = geoTrans[1];

        double y0 = geoTrans[3] + geoTrans[5] / 2.0;
        double dy = geoTrans[5];

        double errX = 0.0;
        double errY = 0.0;
        double maxErrX = 0.0;
        double maxErrY = 0.0;
        double gridLat;
        double gridLon;

        std::valarray<double> geoX(length * width);
        std::valarray<double> geoY(length * width);

        xRaster.getBlock(geoX, 0, 0, width, length);
        yRaster.getBlock(geoY, 0, 0, width, length);

        double square_sum_x = 0; // sum of square differences
        int nvalid_x = 0;
        double square_sum_y = 0; // sum of square differences
        int nvalid_y = 0;

        for (size_t line = 0; line < length; ++line) {
            for (size_t pixel = 0; pixel < width; ++pixel) {
                size_t index = line * width + pixel;
                if (!isnan(geoX[index])) {
                    gridLon = x0 + pixel * dx;
                    errX = geoX[index] - gridLon;
                    square_sum_x += pow(errX, 2);
                    nvalid_x++;
                    if (std::abs(errX) > maxErrX) {
                        maxErrX = std::abs(errX);
                    }
                }
                if (!isnan(geoY[index])) {
                    gridLat = y0 + line * dy;
                    errY = geoY[index] - gridLat;
                    square_sum_y += pow(errY, 2);
                    nvalid_y++;
                    if (std::abs(errY) > maxErrY) {
                        maxErrY = std::abs(errY);
                    }
                }
            }
        }

        double rmse_x = std::sqrt(square_sum_x / nvalid_x);
        double rmse_y = std::sqrt(square_sum_y / nvalid_y);

        std::cout << "geocode_mode: " << geocode_mode_str << std::endl;
        std::cout << "  nvalid X: " << nvalid_x << std::endl;
        std::cout << "  nvalid Y: " << nvalid_y << std::endl;
        std::cout << "  RMSE X: " << rmse_x << std::endl;
        std::cout << "  RMSE Y: " << rmse_y << std::endl;
        std::cout << "  maxErrX: " << maxErrX << std::endl;
        std::cout << "  maxErrY: " << maxErrY << std::endl;
        std::cout << "  dx: " << dx << std::endl;
        std::cout << "  dy: " << dy << std::endl;
    
        ASSERT_GE(nvalid_x, 800);
        ASSERT_GE(nvalid_y, 800);

        if (geocode_mode_str == "interp") {
            // errors with interp algorithm are smaller because topo
            // interpolates x and y at the center of the pixel
            ASSERT_LT(maxErrX, 1.0e-8);
            ASSERT_LT(maxErrY, 1.0e-8);
        }

        ASSERT_LT(rmse_x, 0.5 * dx);
        ASSERT_LT(rmse_y, 0.5 * std::abs(dy));
    }
}

TEST(GeocodeTest, TestGeocodeSlc)
{

    createZeroDem();

    createTestData();

    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);
    std::cout << "H5 opened" << std::endl;

    // Load the product
    std::cout << "create the product" << std::endl;
    isce3::product::Product product(file);

    // std::cout << "get the swath" << std::endl;
    // const isce3::product::Swath & swath = product.swath('A');
    isce3::core::Orbit orbit = product.metadata().orbit();

    std::cout << "construct the ellipsoid" << std::endl;
    isce3::core::Ellipsoid ellipsoid;

    std::cout << "get Doppler" << std::endl;
    // This test relies on that SLC test data in the repo to compute
    // lat, lon, height. In the simulation however I have not added any
    // Carrier so the simulated SLC phase is zero Doppler but its grid is
    // native Doppler. accordingly we can setup the Dopplers as follows.
    // In future we may want to simulate an SLC which has Carrier
    isce3::core::LUT2d<double> imageGridDoppler =
            product.metadata().procInfo().dopplerCentroid('A');

    // construct a zero 2D LUT
    isce3::core::Matrix<double> M(imageGridDoppler.length(),
                                  imageGridDoppler.width());

    M.zeros();
    isce3::core::LUT2d<double> nativeDoppler(
            imageGridDoppler.xStart(), imageGridDoppler.yStart(),
            imageGridDoppler.xSpacing(), imageGridDoppler.ySpacing(), M);

    double thresholdGeo2rdr = 1.0e-9;
    int numiterGeo2rdr = 25;
    size_t linesPerBlock = 1000;
    double demBlockMargin = 0.1;

    // input radar grid
    char freq = 'A';
    std::cout << "construct radar grid" << std::endl;
    isce3::product::RadarGridParameters radarGrid(product, freq);

    double geoGridStartX = -115.65;
    double geoGridStartY = 34.84;
    double geoGridSpacingX = 0.0002;
    double geoGridSpacingY = -8.0e-5;
    int geoGridLength = 500;
    int geoGridWidth = 500;
    int epsgcode = 4326;

    std::cout << "Geogrid" << std::endl;
    isce3::product::GeoGridParameters geoGrid(
            geoGridStartX, geoGridStartY, geoGridSpacingX, geoGridSpacingY,
            geoGridWidth, geoGridLength, epsgcode);

    isce3::io::Raster demRaster("zero_height_dem_geo.bin");

    isce3::io::Raster inputSlc("xslc_rdr.bin", GA_ReadOnly);

    isce3::io::Raster geocodedSlc("xslc_geo.bin", geoGridWidth, geoGridLength, 1,
                                 GDT_CFloat32, "ENVI");

    bool flatten = false;

    isce3::geocode::geocodeSlc(geocodedSlc, inputSlc, demRaster, radarGrid,
                              geoGrid, orbit, nativeDoppler, imageGridDoppler,
                              ellipsoid, thresholdGeo2rdr, numiterGeo2rdr,
                              linesPerBlock, demBlockMargin, flatten);

    isce3::io::Raster inputSlcY("yslc_rdr.bin", GA_ReadOnly);

    isce3::io::Raster geocodedSlcY("yslc_geo.bin", geoGridWidth, geoGridLength, 1,
                                  GDT_CFloat32, "ENVI");

    isce3::geocode::geocodeSlc(geocodedSlcY, inputSlcY, demRaster, radarGrid,
                              geoGrid, orbit, nativeDoppler, imageGridDoppler,
                              ellipsoid, thresholdGeo2rdr, numiterGeo2rdr,
                              linesPerBlock, demBlockMargin, flatten);

    double* _geoTrans = new double[6];
    _geoTrans[0] = geoGridStartX;
    _geoTrans[1] = geoGridSpacingX;
    _geoTrans[2] = 0.0;
    _geoTrans[3] = geoGridStartY;
    _geoTrans[4] = 0.0;
    _geoTrans[5] = geoGridSpacingY;
    geocodedSlcY.setGeoTransform(_geoTrans);
    geocodedSlc.setGeoTransform(_geoTrans);
}

TEST(GeocodeTest, CheckGeocodeSlcResults)
{
    // The geocoded latitude and longitude data should be
    // consistent with the geocoded pixel location.

    isce3::io::Raster xRaster("xslc_geo.bin");

    isce3::io::Raster yRaster("yslc_geo.bin");

    double* geoTrans = new double[6];
    xRaster.getGeoTransform(geoTrans);

    double x0 = geoTrans[0] + geoTrans[1] / 2.0;
    double dx = geoTrans[1];

    double y0 = geoTrans[3] + geoTrans[5] / 2.0;
    double dy = geoTrans[5];

    double deg2rad = M_PI / 180.0;
    x0 *= deg2rad;
    dx *= deg2rad;

    y0 *= deg2rad;
    dy *= deg2rad;

    size_t length = xRaster.length();
    size_t width = xRaster.width();

    std::valarray<std::complex<double>> geoX(length * width);
    std::valarray<std::complex<double>> geoY(length * width);

    xRaster.getBlock(geoX, 0, 0, width, length);

    yRaster.getBlock(geoY, 0, 0, width, length);

    double errX = 0.0;
    double errY = 0.0;
    double maxErrX = 0.0;
    double maxErrY = 0.0;
    double gridLat;
    double gridLon;
    for (size_t line = 0; line < length; ++line) {
        for (size_t pixel = 0; pixel < width; ++pixel) {
            if (std::arg(geoX[line * width + pixel]) != 0.0) {
                gridLon = x0 + pixel * dx;
                errX = std::arg(geoX[line * width + pixel]) - gridLon;

                gridLat = y0 + line * dy;
                errY = std::arg(geoY[line * width + pixel]) - gridLat;

                if (std::abs(errX) > maxErrX) {
                    maxErrX = std::abs(errX);
                }
                if (std::abs(errY) > maxErrY) {
                    maxErrY = std::abs(errY);
                }
            }
        }
    }

    ASSERT_LT(maxErrX, 1.0e-5);
    ASSERT_LT(maxErrY, 1.0e-5);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

void createZeroDem()
{

    // Raster for the existing DEM
    isce3::io::Raster demRaster(TESTDATA_DIR "srtm_cropped.tif");

    // A pointer array for geoTransform
    double geoTrans[6];

    // store the DEM's GeoTransform
    demRaster.getGeoTransform(geoTrans);

    // create a new Raster same as the demRAster
    isce3::io::Raster zeroDemRaster("zero_height_dem_geo.bin", demRaster);
    zeroDemRaster.setGeoTransform(geoTrans);
    zeroDemRaster.setEPSG(demRaster.getEPSG());

    size_t length = demRaster.length();
    size_t width = demRaster.width();

    std::valarray<float> dem(length * width);
    dem = 0.0;
    zeroDemRaster.setBlock(dem, 0, 0, width, length);
}

void createTestData()
{

    // Open the HDF5 product
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Load the product
    isce3::product::Product product(file);

    // Create topo instance with native Doppler
    isce3::geometry::Topo topo(product, 'A', true);

    // Load topo processing parameters to finish configuration
    topo.threshold(0.05);
    topo.numiter(25);
    topo.extraiter(10);
    topo.demMethod(isce3::core::dataInterpMethod::BIQUINTIC_METHOD);
    topo.epsgOut(4326);

    // Open DEM raster
    isce3::io::Raster demRaster("zero_height_dem_geo.bin");

    // Run topo
    topo.topo(demRaster, ".");

    isce3::io::Raster xRaster("x.rdr");
    isce3::io::Raster yRaster("y.rdr");

    size_t length = xRaster.length();
    size_t width = xRaster.width();

    std::valarray<double> x(width * length);
    std::valarray<std::complex<float>> xslc(width * length);
    xRaster.getBlock(x, 0, 0, width, length);
    x *= M_PI / 180.0;

    std::valarray<double> y(width * length);
    std::valarray<std::complex<float>> yslc(width * length);
    yRaster.getBlock(y, 0, 0, width, length);
    y *= M_PI / 180.0;

    std::valarray<std::complex<float>> x_conj_y_slc(width * length);

    for (int ii = 0; ii < width * length; ++ii) {

        const std::complex<float> cpxPhaseX(std::cos(x[ii]), std::sin(x[ii]));
        xslc[ii] = cpxPhaseX;

        const std::complex<float> cpxPhaseY(std::cos(y[ii]), std::sin(y[ii]));
        yslc[ii] = cpxPhaseY;

        x_conj_y_slc[ii] = cpxPhaseX * std::conj(cpxPhaseY);
    }

    isce3::io::Raster slcRasterX("xslc_rdr.bin", width, length, 1, GDT_CFloat32,
                                "ENVI");

    slcRasterX.setBlock(xslc, 0, 0, width, length);

    isce3::io::Raster slcRasterY("yslc_rdr.bin", width, length, 1, GDT_CFloat32,
                                "ENVI");

    slcRasterY.setBlock(yslc, 0, 0, width, length);

    isce3::io::Raster slc_x_conj_y_raster("x_conj_y_slc_rdr.bin", width, length, 1,
                                      GDT_CFloat32, "ENVI");

    slc_x_conj_y_raster.setBlock(x_conj_y_slc, 0, 0, width, length);
}
