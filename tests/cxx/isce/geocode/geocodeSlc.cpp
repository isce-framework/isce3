#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>

#include <isce/core/Ellipsoid.h>
#include <isce/core/LUT2d.h>
#include <isce/core/Metadata.h>
#include <isce/core/Orbit.h>
#include <isce/geocode/geocodeSlc.h>
#include <isce/geometry/Serialization.h>
#include <isce/geometry/Topo.h>
#include <isce/io/IH5.h>
#include <isce/io/Raster.h>
#include <isce/product/GeoGridParameters.h>
#include <isce/product/Product.h>
#include <isce/product/Serialization.h>

// Declaration for utility function to read metadata stream from V  RT
std::stringstream streamFromVRT(const char* filename, int bandNum = 1);

// To create a zero height DEM
void createZeroDem();

// To create test data
void createTestData();

TEST(geocodeTest, TestGeocodeSlc)
{

    createZeroDem();

    createTestData();

    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce::io::IH5File file(h5file);
    std::cout << "H5 opened" << std::endl;

    // Load the product
    std::cout << "create the product" << std::endl;
    isce::product::Product product(file);

    // std::cout << "get the swath" << std::endl;
    // const isce::product::Swath & swath = product.swath('A');
    isce::core::Orbit orbit = product.metadata().orbit();

    std::cout << "construct the ellipsoid" << std::endl;
    isce::core::Ellipsoid ellipsoid;

    std::cout << "get Doppler" << std::endl;
    // This test relies on that SLC test data in the repo to compute
    // lat, lon, height. In the simulation however I have not added any
    // Carrier so the simulated SLC phase is zero Doppler but its grid is
    // native Doppler. accordingly we can setup the Dopplers as follows.
    // In future we may want to simulate an SLC which has Carrier
    isce::core::LUT2d<double> imageGridDoppler =
            product.metadata().procInfo().dopplerCentroid('A');

    // construct a zero 2D LUT
    isce::core::Matrix<double> M(imageGridDoppler.width(),
                                 imageGridDoppler.length());

    M.zeros();
    isce::core::LUT2d<double> nativeDoppler(
            imageGridDoppler.xStart(), imageGridDoppler.yStart(),
            imageGridDoppler.xSpacing(), imageGridDoppler.ySpacing(), M);

    // auto lookSide = product.lookSide();

    double thresholdGeo2rdr = 1.0e-9;
    int numiterGeo2rdr = 25;
    size_t linesPerBlock = 1000;
    double demBlockMargin = 0.1;

    // input radar grid
    char freq = 'A';
    std::cout << "construct radar grid" << std::endl;
    isce::product::RadarGridParameters radarGrid(product, freq);

    double geoGridStartX = -115.65;
    double geoGridStartY = 34.84;
    double geoGridSpacingX = 0.0002;
    double geoGridSpacingY = -8.0e-5;
    int geoGridLength = 500;
    int geoGridWidth = 500;
    int epsgcode = 4326;

    std::cout << "Geogrid" << std::endl;
    isce::product::GeoGridParameters geoGrid(
            geoGridStartX, geoGridStartY, geoGridSpacingX, geoGridSpacingY,
            geoGridWidth, geoGridLength, epsgcode);

    isce::io::Raster demRaster("zeroHeightDEM.geo");

    isce::io::Raster inputSlc("x.slc", GA_ReadOnly);

    isce::io::Raster geocodedSlc("xslc.geo", geoGridWidth, geoGridLength, 1,
                                 GDT_CFloat32, "ENVI");

    bool flatten = false;

    isce::geocode::geocodeSlc(geocodedSlc, inputSlc, demRaster, radarGrid,
                              geoGrid, orbit, nativeDoppler, imageGridDoppler,
                              ellipsoid, thresholdGeo2rdr, numiterGeo2rdr,
                              linesPerBlock, demBlockMargin, flatten);

    isce::io::Raster inputSlcY("y.slc", GA_ReadOnly);

    isce::io::Raster geocodedSlcY("yslc.geo", geoGridWidth, geoGridLength, 1,
                                  GDT_CFloat32, "ENVI");

    isce::geocode::geocodeSlc(geocodedSlcY, inputSlcY, demRaster, radarGrid,
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

TEST(GeocodeTest, CheckGeocode)
{
    // The geocoded latitude and longitude data should be
    // consistent with the geocoded pixel location.

    isce::io::Raster xRaster("xslc.geo");

    isce::io::Raster yRaster("yslc.geo");

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
    isce::io::Raster demRaster(TESTDATA_DIR "srtm_cropped.tif");

    // A pointer array for geoTransform
    double geoTrans[6];

    // store the DEM's GeoTransform
    demRaster.getGeoTransform(geoTrans);

    // create a new Raster same as the demRAster
    isce::io::Raster zeroDemRaster("zeroHeightDEM.geo", demRaster);
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
    isce::io::IH5File file(h5file);

    // Load the product
    isce::product::Product product(file);

    isce::core::LUT2d<double> nativeDoppler =
            product.metadata().procInfo().dopplerCentroid('A');
    isce::core::LUT2d<double> imageGridDoppler =
            product.metadata().procInfo().dopplerCentroid('A');

    // Create topo instance with native Doppler
    isce::geometry::Topo topo(product, 'A', true);

    // Load topo processing parameters to finish configuration
    std::ifstream xmlfid(TESTDATA_DIR "topo.xml", std::ios::in);
    {
        cereal::XMLInputArchive archive(xmlfid);
        archive(cereal::make_nvp("Topo", topo));
    }

    // Open DEM raster
    isce::io::Raster demRaster("zeroHeightDEM.geo");

    // Run topo
    topo.topo(demRaster, ".");

    isce::io::Raster xRaster("x.rdr");
    isce::io::Raster yRaster("y.rdr");

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

    isce::io::Raster slcRasterX("x.slc", width, length, 1, GDT_CFloat32,
                                "ENVI");

    slcRasterX.setBlock(xslc, 0, 0, width, length);

    isce::io::Raster slcRasterY("y.slc", width, length, 1, GDT_CFloat32,
                                "ENVI");

    slcRasterY.setBlock(yslc, 0, 0, width, length);
}
