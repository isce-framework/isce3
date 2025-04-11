#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <optional>

#include <gtest/gtest.h>

#include <isce3/core/Ellipsoid.h>
#include <isce3/core/EMatrix.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Metadata.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly2d.h>
#include <isce3/geocode/geocodeSlc.h>
#include <isce3/geocode/GeocodeCov.h>
#include <isce3/geometry/Topo.h>
#include <isce3/io/IH5.h>
#include <isce3/io/Raster.h>
#include <isce3/math/Stats.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/product/RadarGridProduct.h>
#include <isce3/product/Serialization.h>
#include <isce3/product/SubSwaths.h>

using isce3::math::computeRasterStats;

// Declaration for utility function to read metadata stream from V  RT
std::stringstream streamFromVRT(const char* filename, int bandNum = 1);

// To create a zero height DEM
void createZeroDem();

// To create test data
void createTestData();

// global geocode SLC modes shared between running and checking
std::set<std::string> axes = {"x", "y"};
std::set<std::string> offset_modes = {"", "_rg", "_az", "_rg_az"};
std::set<std::string> gslc_modes = {"_raster", "_array"};

TEST(GeocodeTest, TestGeocodeSlc)
{
    createZeroDem();
    createTestData();

    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);
    std::cout << "H5 opened" << std::endl;

    // Load the product
    std::cout << "create the product" << std::endl;
    isce3::product::RadarGridProduct product(file);

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

    // input radar grid (baseline - will be copied and altered as neeed)
    char freq = 'A';
    std::cout << "construct radar grid" << std::endl;
    isce3::product::RadarGridParameters radarGrid(product, freq);

    // common geogrid
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

    // common geotrans to be applied to all rasters
    double* _geoTrans = new double[6];
    _geoTrans[0] = geoGridStartX;
    _geoTrans[1] = geoGridSpacingX;
    _geoTrans[2] = 0.0;
    _geoTrans[3] = geoGridStartY;
    _geoTrans[4] = 0.0;
    _geoTrans[5] = geoGridSpacingY;

    // common carrier default carrier LUT2d
    const auto default_carrier_lut2d = isce3::core::LUT2d<double>();

    // common default correction LUT2d
    const auto default_correction_lut2d = isce3::core::LUT2d<double>();

    // multiplicative factor applied to range pixel spacing and azimuth time
    // interval to be added to starting range and azimuth time of radar grid
    const double offset_factor = 10.0;

    // create azimuth correction LUT2d with matrix fill with azimuth time
    // interval (1/PRF) multiplied by offset factor to amplify effect
    isce3::core::Matrix<double> m_az_correct(radarGrid.length(),
                                             radarGrid.width());
    const auto az_time_interval = 1 / radarGrid.prf();
    m_az_correct.fill(offset_factor * az_time_interval);

    // create range correction LUT2d with matrix filled with range pixel
    // spacing multiplied by offset factor to amplify effect
    isce3::core::Matrix<double> m_srange_correct(radarGrid.length(),
                                                 radarGrid.width());
    m_srange_correct.fill(offset_factor * radarGrid.rangePixelSpacing());

    // common DEM raster
    isce3::io::Raster demRaster("zero_height_dem_geo.bin");

    // init output array (will be reused in all geocodeSlc array mode tests)
    isce3::core::EArray2D<std::complex<float>>
        geoDataArr(geoGridLength, geoGridWidth);

    bool flatten = false;
    bool reramp = true;

    // iterate over different axes and corrections, and geocode
    for (auto axis : axes) {
        // input radar raster
        isce3::io::Raster inputSlc(axis + "slc_rdr.bin", GA_ReadOnly);

        // input radar raster as array
        isce3::core::EArray2D<std::complex<float>> rdrDataArr(
                inputSlc.length(), inputSlc.width());
        inputSlc.getBlock(rdrDataArr.data(), 0, 0, inputSlc.width(),
                inputSlc.length(), 1);

        for (auto offset_mode : offset_modes) {
            // test radar grid to be altered as needed
            auto testRdrGrid = radarGrid;

            // az time correction LUT2d and radar grid based on offset mode
            isce3::core::LUT2d<double> az_correction = default_correction_lut2d;
            if (offset_mode.find("az") != std::string::npos) {
                testRdrGrid.sensingStart(radarGrid.sensingStart()
                        + offset_factor * az_time_interval);
                az_correction = isce3::core::LUT2d<double>(
                        testRdrGrid.startingRange(),
                        testRdrGrid.sensingStart(),
                        testRdrGrid.rangePixelSpacing(), az_time_interval,
                        m_az_correct);
            }

            // range correction LUT2d and radar grid based on offset mode
            isce3::core::LUT2d<double> srange_correction = default_correction_lut2d;
            if (offset_mode.find("rg") != std::string::npos) {
                testRdrGrid.startingRange(radarGrid.startingRange()
                        + offset_factor * radarGrid.rangePixelSpacing());
                srange_correction = isce3::core::LUT2d<double>(
                        testRdrGrid.startingRange(),
                        testRdrGrid.sensingStart(),
                        testRdrGrid.rangePixelSpacing(), az_time_interval,
                        m_srange_correct);
            }

            // output name common to both raster and array geocodeSlc modes
            const std::string filePrefix = axis + "slc_geo" + offset_mode;

            // geocde SLC in raster mode
            isce3::io::Raster geocodedSlcRaster(filePrefix +  "_raster.bin",
                    geoGridWidth, geoGridLength, 1, GDT_CFloat32, "ENVI");
            isce3::geocode::geocodeSlc(geocodedSlcRaster, inputSlc, demRaster,
                    testRdrGrid, geoGrid, orbit, nativeDoppler, imageGridDoppler,
                    ellipsoid, thresholdGeo2rdr, numiterGeo2rdr, linesPerBlock,
                    flatten, reramp, default_carrier_lut2d,
                    default_carrier_lut2d, az_correction, srange_correction);
            geocodedSlcRaster.setGeoTransform(_geoTrans);

            // insert references to geo and radar arrays in a vector
            std::vector<isce3::geocode::EArray2dc64> geoDataVec = {geoDataArr};
            std::vector<isce3::geocode::EArray2dc64> rdrDataVec = {rdrDataArr};

            // create reference to mask array
            auto maskArr2d = isce3::core::EArray2D<unsigned char>(geoGridLength,
                                                                  geoGridWidth);
            maskArr2d.fill(0);
            auto maskArr2dRef = isce3::geocode::EArray2duc8(maskArr2d);
            auto maskArr2dRefOpt = std::make_optional(maskArr2dRef);

            // create empty array for carrier and flattening phases
            // empty arrays do not affect processing
            auto dummy = isce3::core::EArray2D<double>();

            // geocodeSlc in array mode and write array to raster
            isce3::geocode::geocodeSlc(geoDataVec, dummy, dummy,
                    rdrDataVec, demRaster, radarGrid, radarGrid, geoGrid, orbit,
                    nativeDoppler, imageGridDoppler, ellipsoid,
                    thresholdGeo2rdr, numiterGeo2rdr,
                    maskArr2dRefOpt,
                    0, 0, flatten, reramp);
            isce3::io::Raster geocodedSlcArr(filePrefix + "_array.bin",
                    geoGridWidth, geoGridLength, 1, GDT_CFloat32, "ENVI");
            geocodedSlcArr.setBlock(geoDataArr.data(), 0, 0, geoGridWidth,
                    geoGridLength, 1);
            geocodedSlcArr.setGeoTransform(_geoTrans);
        } // loop over offset_modes
    } // loop over axes
}

TEST(GeocodeTest, CheckGeocodeSlc)
{
    // The geocoded latitude and longitude data should be
    // consistent with the geocoded pixel location.

    double* geoTrans = new double[6];
    double err;
    double gridVal;

    // iterate over different axes and corrections, and track number of times
    // max error threshold is exceeded
    size_t nFails = 0;
    for (auto axis : axes) {
        for (auto offset_mode : offset_modes) {
            for (auto gslc_mode : gslc_modes) {
                // open current test output raster and load output to array
                std::string fileName = axis + "slc_geo" + offset_mode + gslc_mode + ".bin";
                isce3::io::Raster geoRaster(fileName);
                const auto length = geoRaster.length();
                const auto width = geoRaster.width();
                std::valarray<std::complex<double>> geoData(length * width);
                geoRaster.getBlock(geoData, 0, 0, width, length);

                // use geotransfrom in geo raster to init
                geoRaster.getGeoTransform(geoTrans);
                double x0 = geoTrans[0] + geoTrans[1] / 2.0;
                double dx = geoTrans[1];

                double y0 = geoTrans[3] + geoTrans[5] / 2.0;
                double dy = geoTrans[5];

                double deg2rad = M_PI / 180.0;
                x0 *= deg2rad;
                dx *= deg2rad;

                y0 *= deg2rad;
                dy *= deg2rad;

                // max error of current raster
                double maxErr = 0.0;

                // loop over lines and pixels of output and check output
                for (size_t line = 0; line < length; ++line) {
                    for (size_t pixel = 0; pixel < width; ++pixel) {
                        // skip invalid pixels
                        if (isnan(std::real(geoData[line * width + pixel])))
                            continue;

                        // compute expected grid value based on axis
                        gridVal = (axis == "x") ? x0 + pixel * dx : y0 + line * dy;

                        // compute error and check if it's max
                        err = std::arg(geoData[line * width + pixel]) - gridVal;
                        maxErr = std::max(maxErr, std::abs(err));
                    } // loop over pixel
                } // loop over line

                // increment fails if maxErr greather than threshold
                if (maxErr > 1.0e-6) {
                    nFails++;
                }
                std::cout << fileName << "\t" << maxErr << std::endl;
            } // loop over gslc modes
        } // loop over offset modes
    } // loop over axes

    ASSERT_EQ(nFails, 0);
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
    isce3::product::RadarGridProduct product(file);

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

    // init raster containing longitude (degrees) of each radar grid pixel
    // will be treated as phase later
    isce3::io::Raster xRaster("x.rdr");
    // init raster containing latitude (degrees) of each radar grid pixel
    // will be treated as phase later
    isce3::io::Raster yRaster("y.rdr");

    // get raster dims
    size_t length = xRaster.length();
    size_t width = xRaster.width();

    // load longitude values as radians from raster
    std::valarray<double> x(width * length);
    xRaster.getBlock(x, 0, 0, width, length);
    x *= M_PI / 180.0;

    // load latitude values as radians from raster
    std::valarray<double> y(width * length);
    yRaster.getBlock(y, 0, 0, width, length);
    y *= M_PI / 180.0;

    // output array for longitude as complex/SLC
    std::valarray<std::complex<float>> xslc(width * length);
    // output array for latitude as complex/SLC
    std::valarray<std::complex<float>> yslc(width * length);
    // output array for complex lon * conjugate of complex lat
    // for geocodeCov area proj testing
    std::valarray<std::complex<float>> x_conj_y_slc(width * length);

    for (int ii = 0; ii < width * length; ++ii) {

        // save longitude value as complex value
        const std::complex<float> cpxPhaseX(std::cos(x[ii]), std::sin(x[ii]));
        xslc[ii] = cpxPhaseX;

        // save latitude value as complex value
        const std::complex<float> cpxPhaseY(std::cos(y[ii]), std::sin(y[ii]));
        yslc[ii] = cpxPhaseY;

        // save product of complex lon and conjugate of complex lat
        x_conj_y_slc[ii] = cpxPhaseX * std::conj(cpxPhaseY);
    }

    // write SLCs to disk
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
