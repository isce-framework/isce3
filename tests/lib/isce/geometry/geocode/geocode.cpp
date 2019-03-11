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

#include "isce/io/Raster.h"
#include <isce/io/IH5.h>
#include <isce/product/Serialization.h>

#include <isce/core/Metadata.h>
#include <isce/core/Orbit.h>
#include <isce/core/Poly2d.h>
#include <isce/core/LUT1d.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Projections.h>
#include <isce/core/Interpolator.h>
#include <isce/core/Constants.h>

// isce::geometry
#include "isce/geometry/Serialization.h"
#include "isce/geometry/Topo.h"

#include <isce/geometry/Geocode.h>

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

    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Load the product
    isce::product::Product product(file);
    
    const isce::product::Swath & swath = product.swath('A');
    isce::core::Orbit orbit = product.metadata().orbit();
    isce::core::Ellipsoid ellipsoid;
    isce::core::LUT2d<double> doppler = product.metadata().procInfo().dopplerCentroid('A');

    double threshold = 1.0e-9 ;
    int numiter = 25;
    size_t linesPerBlock = 1000;
    double demBlockMargin = 0.1;
    int radarBlockMargin = 10;

    double azimuthTimeInterval = 1.0/swath.nominalAcquisitionPRF();

    // output geocoded grid (can be different from DEM)
    double geoGridStartX = -115.65;
    double geoGridStartY = 34.84;
    double geoGridSpacingX = 0.0002;
    double geoGridSpacingY = -8.0e-5;
    int geoGridLength = 500;
    int geoGridWidth = 500;
    int epsgcode = 4326;

    // The DEM to be used for geocoding 
    isce::io::Raster demRaster("zeroHeightDEM.geo");

    // input raster in radar coordinates to be geocoded
    isce::io::Raster radarRaster("x.rdr");

    // geocoded raster
    isce::io::Raster geocodedRaster("x.geo", 
                        geoGridWidth, geoGridLength,
                        1, GDT_Float64, "ENVI");

    int radarGridLength = radarRaster.length();
    int radarGridWidth = radarRaster.width();

    // The interpolation method used for geocoding
    //isce::core::dataInterpMethod method = isce::core::BICUBIC_METHOD;
    //isce::core::dataInterpMethod method = isce::core::BILINEAR_METHOD;
    isce::core::dataInterpMethod method = isce::core::BIQUINTIC_METHOD;

    // Geocode object
    isce::geometry::Geocode<double> geoObj;

    // manually configure geoObj
    
    geoObj.orbit(orbit);

    geoObj.ellipsoid(ellipsoid);

    geoObj.thresholdGeo2rdr(threshold);

    geoObj.numiterGeo2rdr(numiter);

    geoObj.linesPerBlock(linesPerBlock);

    geoObj.demBlockMargin(demBlockMargin);

    geoObj.radarBlockMargin(radarBlockMargin);

    geoObj.interpolator(method);

    geoObj.radarGrid(doppler,
                      orbit.refEpoch,
                      swath.zeroDopplerTime()[0],
                      azimuthTimeInterval,
                      radarGridLength,
                      swath.slantRange()[0],
                      swath.rangePixelSpacing(),
                      swath.processedWavelength(),
                      radarGridWidth);

    geoObj.geoGrid(geoGridStartX, geoGridStartY,
                  geoGridSpacingX, geoGridSpacingY,
                  geoGridWidth, geoGridLength,
                  epsgcode);

    // geocode the longitude data
    geoObj.geocode(radarRaster, geocodedRaster, demRaster);

    // create another raster for latitude data from Topo
    isce::io::Raster radarRaster2("y.rdr");

    // create output raster for geocoded latitude
    isce::io::Raster geocodedRaster2("y.geo", geoGridWidth, geoGridLength,
                1, GDT_Float64, "ENVI");

    // geocode the latitude data using the same geocode object
    geoObj.geocode(radarRaster2, geocodedRaster2, demRaster);

}

TEST(GeocodeTest, CheckGeocode) {
    // The geocoded latitude and longitude data should be 
    // consistent with the geocoded pixel location.
    
    isce::io::Raster xRaster("x.geo");

    isce::io::Raster yRaster("y.geo");

    double * geoTrans = new double[6];
    xRaster.getGeoTransform(geoTrans);
    
    double x0 = geoTrans[0] + geoTrans[1]/2.0;
    double dx = geoTrans[1];

    double y0 = geoTrans[3] + geoTrans[5]/2.0;
    double dy = geoTrans[5];

    size_t length = xRaster.length();
    size_t width = xRaster.width();

    std::valarray<double> geoX(length*width);
    std::valarray<double> geoY(length*width);

    xRaster.getBlock(geoX, 0 ,0 , width, length);

    yRaster.getBlock(geoY, 0 ,0 , width, length);

    double errX = 0.0;
    double errY = 0.0;
    double maxErrX = 0.0;
    double maxErrY = 0.0;
    double gridLat;
    double gridLon;
    for (size_t line = 0; line < length; ++line) {
        for (size_t pixel = 0; pixel < width; ++pixel) {
            if (geoX[line*width + pixel] != 0.0) {
                gridLon = x0 + pixel * dx;
                errX = geoX[line*width + pixel] - gridLon;
                
                gridLat = y0 + line * dy;
                errY = geoY[line*width + pixel] - gridLat;

                if (std::abs(errX) > maxErrX){
                    maxErrX = std::abs(errX);
                }

                if (std::abs(errY) > maxErrY){
                    maxErrY = std::abs(errY);
                }

            }
        }

    }

    ASSERT_LT(maxErrX, 1.0e-8);
    ASSERT_LT(maxErrY, 1.0e-8);

}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

void createZeroDem() {

    // Raster for the existing DEM
    isce::io::Raster demRaster("../../data/srtm_cropped.tif");

    // A pointer array for geoTransform
    double * geoTrans = new double[6]; 

    // store the DEM's GeoTransform
    demRaster.getGeoTransform(geoTrans);

    // create a new Raster same as the demRAster
    isce::io::Raster zeroDemRaster("zeroHeightDEM.geo", demRaster);
    zeroDemRaster.setGeoTransform(geoTrans);

    size_t length = demRaster.length();
    size_t width = demRaster.width();

    std::valarray<float> dem(length*width);
    dem = 0.0;
    zeroDemRaster.setBlock(dem, 0, 0, width, length);

}

void createTestData() {

    // Open the HDF5 product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Load the product
    isce::product::Product product(file);

    // Create topo instance with native Doppler
    isce::geometry::Topo topo(product, 'A', true);

    // Load topo processing parameters to finish configuration
    std::ifstream xmlfid("../../data/topo.xml", std::ios::in);
    {
    cereal::XMLInputArchive archive(xmlfid);
    archive(cereal::make_nvp("Topo", topo));
    }

    // Open DEM raster
    isce::io::Raster demRaster("zeroHeightDEM.geo");

    // Run topo
    topo.topo(demRaster, ".");

}


