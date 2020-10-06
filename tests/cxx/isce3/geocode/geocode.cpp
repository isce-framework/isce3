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
#include <isce3/geometry/Serialization.h>
#include <isce3/geometry/Topo.h>
#include <isce3/io/IH5.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/product/Product.h>
#include <isce3/product/Serialization.h>

std::set<std::string> geocode_mode_set = {"interp", "areaProj"};

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
    isce3::io::Raster demRaster("zeroHeightDEM.geo");

    // input raster in radar coordinates to be geocoded
    isce3::io::Raster radarRasterX("x.rdr");

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

    for (auto geocode_mode_str : geocode_mode_set) {

        std::cout << "geocode_mode: " << geocode_mode_str << std::endl;

        isce3::geocode::geocodeOutputMode output_mode;
        if (geocode_mode_str == "interp")
            output_mode = isce3::geocode::geocodeOutputMode::INTERP;
        else
            output_mode = isce3::geocode::geocodeOutputMode::AREA_PROJECTION;

        // geocoded raster
        isce3::io::Raster geocodedRasterInterpX("x." + geocode_mode_str + ".geo",
                                               geoGridWidth, geoGridLength, 1,
                                               GDT_Float64, "ENVI");

        // geocode the longitude data
        geoObj.geocode(radar_grid, radarRasterX, geocodedRasterInterpX,
                       demRaster, output_mode);
    }

    for (auto geocode_mode_str : geocode_mode_set) {

        isce3::geocode::geocodeOutputMode output_mode;
        if (geocode_mode_str == "interp")
            output_mode = isce3::geocode::geocodeOutputMode::INTERP;
        else
            output_mode = isce3::geocode::geocodeOutputMode::AREA_PROJECTION;

        // create another raster for latitude data from Topo
        isce3::io::Raster radarRasterY("y.rdr");

        // create output raster for geocoded latitude
        isce3::io::Raster geocodedRasterInterpY("y." + geocode_mode_str + ".geo",
                                               geoGridWidth, geoGridLength, 1,
                                               GDT_Float64, "ENVI");

        // geocode the latitude data using the same geocode object
        geoObj.geocode(radar_grid, radarRasterY, geocodedRasterInterpY,
                       demRaster, output_mode);
    }
}

TEST(GeocodeTest, CheckGeocodeCovResults) {
    // The geocoded latitude and longitude data should be
    // consistent with the geocoded pixel location.

    for (auto geocode_mode_str : geocode_mode_set) {

        isce3::io::Raster xRaster("x." + geocode_mode_str + ".geo");
        isce3::io::Raster yRaster("y." + geocode_mode_str + ".geo");
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
                if (!isnan(geoX[line * width + pixel])) {
                    gridLon = x0 + pixel * dx;
                    errX = geoX[line * width + pixel] - gridLon;
                    square_sum_x += pow(errX, 2);
                    nvalid_x++;
                    if (std::abs(errX) > maxErrX) {
                        maxErrX = std::abs(errX);
                    }
                }
                if (!isnan(geoY[line * width + pixel])) {
                    gridLat = y0 + line * dy;
                    errY = geoY[line * width + pixel] - gridLat;
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
        std::cout << "  RMSE X: " << rmse_x << std::endl;
        std::cout << "  RMSE Y: " << rmse_y << std::endl;
        std::cout << "  maxErrX: " << maxErrX << std::endl;
        std::cout << "  maxErrY: " << maxErrY << std::endl;
        std::cout << "  dx: " << dx << std::endl;
        std::cout << "  dy: " << dy << std::endl;

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

    isce3::io::Raster demRaster("zeroHeightDEM.geo");

    isce3::io::Raster inputSlc("xslc", GA_ReadOnly);

    isce3::io::Raster geocodedSlc("xslc.geo", geoGridWidth, geoGridLength, 1,
                                 GDT_CFloat32, "ENVI");

    bool flatten = false;

    isce3::geocode::geocodeSlc(geocodedSlc, inputSlc, demRaster, radarGrid,
                              geoGrid, orbit, nativeDoppler, imageGridDoppler,
                              ellipsoid, thresholdGeo2rdr, numiterGeo2rdr,
                              linesPerBlock, demBlockMargin, flatten);

    isce3::io::Raster inputSlcY("yslc", GA_ReadOnly);

    isce3::io::Raster geocodedSlcY("yslc.geo", geoGridWidth, geoGridLength, 1,
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

    isce3::io::Raster xRaster("xslc.geo");

    isce3::io::Raster yRaster("yslc.geo");

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
    isce3::io::Raster zeroDemRaster("zeroHeightDEM.geo", demRaster);
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

    isce3::core::LUT2d<double> nativeDoppler =
            product.metadata().procInfo().dopplerCentroid('A');
    isce3::core::LUT2d<double> imageGridDoppler =
            product.metadata().procInfo().dopplerCentroid('A');

    // Create topo instance with native Doppler
    isce3::geometry::Topo topo(product, 'A', true);

    // Load topo processing parameters to finish configuration
    std::ifstream xmlfid(TESTDATA_DIR "topo.xml", std::ios::in);
    {
        cereal::XMLInputArchive archive(xmlfid);
        archive(cereal::make_nvp("Topo", topo));
    }

    // Open DEM raster
    isce3::io::Raster demRaster("zeroHeightDEM.geo");

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

    for (int ii = 0; ii < width * length; ++ii) {

        const std::complex<float> cpxPhaseX(std::cos(x[ii]), std::sin(x[ii]));
        xslc[ii] = cpxPhaseX;

        const std::complex<float> cpxPhaseY(std::cos(y[ii]), std::sin(y[ii]));
        yslc[ii] = cpxPhaseY;
    }

    isce3::io::Raster slcRasterX("xslc", width, length, 1, GDT_CFloat32,
                                "ENVI");

    slcRasterX.setBlock(xslc, 0, 0, width, length);

    isce3::io::Raster slcRasterY("yslc", width, length, 1, GDT_CFloat32,
                                "ENVI");

    slcRasterY.setBlock(yslc, 0, 0, width, length);
}
