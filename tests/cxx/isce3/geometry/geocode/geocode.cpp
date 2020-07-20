//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-
//
#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

#include "isce3/io/Raster.h"
#include <isce3/io/IH5.h>
#include <isce3/product/Serialization.h>

#include <isce3/core/Metadata.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Poly2d.h>
#include <isce3/core/LUT1d.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Projections.h>
#include <isce3/core/Interpolator.h>
#include <isce3/core/Constants.h>

// isce3::geometry
#include "isce3/geometry/Serialization.h"
#include "isce3/geometry/Topo.h"

#include <isce3/geometry/Geocode.h>

std::set<std::string> geocode_mode_set = {"interp", "areaProj"};

// Declaration for utility function to read metadata stream from VRT
std::stringstream streamFromVRT(const char * filename, int bandNum=1);

// To create a zero height DEM
void createZeroDem();

// To create test data
void createTestData();

TEST(GeocodeTest, RunGeocode) {

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
    isce3::geometry::Geocode<double> geoObj;

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

        isce3::geometry::geocodeOutputMode output_mode;
        if (geocode_mode_str == "interp")
            output_mode = isce3::geometry::geocodeOutputMode::INTERP;
        else
            output_mode = isce3::geometry::geocodeOutputMode::AREA_PROJECTION;

        // geocoded raster
        isce3::io::Raster geocodedRasterInterpX("x." + geocode_mode_str + ".geo",
                                               geoGridWidth, geoGridLength, 1,
                                               GDT_Float64, "ENVI");

        // geocode the longitude data
        geoObj.geocode(radar_grid, radarRasterX, geocodedRasterInterpX,
                       demRaster, output_mode);
    }

    for (auto geocode_mode_str : geocode_mode_set) {

        isce3::geometry::geocodeOutputMode output_mode;
        if (geocode_mode_str == "interp")
            output_mode = isce3::geometry::geocodeOutputMode::INTERP;
        else
            output_mode = isce3::geometry::geocodeOutputMode::AREA_PROJECTION;

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

TEST(GeocodeTest, CheckGeocode) {
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

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

void createZeroDem() {

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

    std::valarray<float> dem(length*width);
    dem = 0.0;
    zeroDemRaster.setBlock(dem, 0, 0, width, length);

}

void createTestData() {

    // Open the HDF5 product
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Load the product
    isce3::product::Product product(file);

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

}


