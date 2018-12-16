//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2018
//

#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <gtest/gtest.h>

// isce::io
#include "isce/io/IH5.h"

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/DateTime.h"
#include "isce/core/Ellipsoid.h"
#include "isce/core/Orbit.h"
#include "isce/core/LUT1d.h"
#include "isce/core/Serialization.h"

// isce::product
#include "isce/product/Product.h"

// isce::geometry
#include "isce/geometry/geometry.h"

// Declaration for utility function to read test data
void loadTestData(std::vector<std::string> & aztimes, std::vector<double> & ranges,
                  std::vector<double> & heights,
                  std::vector<double> & ref_data, std::vector<double> & ref_zerodop);

struct GeometryTest : public ::testing::Test {

    // isce::core objects
    isce::core::Ellipsoid ellipsoid;
    isce::core::LUT1d<double> skewDoppler;
    isce::core::Orbit orbit;
    // isce::product objects
    isce::product::ImageMode mode;

    int lookSide;

    // Constructor
    protected:
        GeometryTest() {
            // Open the HDF5 product
            std::string h5file("../../data/envisat.h5");
            isce::io::IH5File file(h5file);

            // Instantiate a Product
            isce::product::Product product(file);

            // Extract core and product objects
            ellipsoid = product.metadata().identification().ellipsoid();
            orbit = product.metadata().orbitPOE();
            skewDoppler = product.metadata().instrument().skewDoppler();
            mode = product.complexImagery().primaryMode();
            lookSide = product.metadata().identification().lookDirection();
        }
};

TEST_F(GeometryTest, RdrToGeoWithOrbit) {
    
    // Load test data
    std::vector<std::string> aztimes;
    std::vector<double> ranges, heights, ref_data, ref_zerodop;
    loadTestData(aztimes, ranges, heights, ref_data, ref_zerodop);

    // Create reference epoch
    isce::core::DateTime azDate(aztimes[0]);
    isce::core::DateTime refEpoch = azDate - 86400*2;
    orbit.updateUTCTimes(refEpoch);

    // Loop over test data
    const double degrees = 180.0 / M_PI;
    for (size_t i = 0; i < aztimes.size(); ++i) {

        // Make azimuth date
        azDate = aztimes[i]; 
        const double azTime = azDate.secondsSinceEpoch(refEpoch);

        // Evaluate Doppler
        const double doppler = skewDoppler.eval(ranges[i]);

        // Make constant DEM interpolator set to input height
        isce::geometry::DEMInterpolator dem(heights[i]);

        // Initialize guess
        isce::core::cartesian_t targetLLH = {0.0, 0.0, dem.interpolateLonLat(0.0, 0.0)};

        // Run rdr2geo
        int stat = isce::geometry::rdr2geo(azTime, ranges[i], doppler,
            orbit, ellipsoid, dem, targetLLH, mode.wavelength(), lookSide,
            1.0e-8, 25, 15, isce::core::HERMITE_METHOD);

        // Check
        ASSERT_EQ(stat, 1);
        ASSERT_NEAR(degrees * targetLLH[0], ref_data[3*i], 1.0e-8);
        ASSERT_NEAR(degrees * targetLLH[1], ref_data[3*i+1], 1.0e-8);
        ASSERT_NEAR(targetLLH[2], ref_data[3*i+2], 1.0e-8);

        // Run again with zero doppler
        stat = isce::geometry::rdr2geo(azTime, ranges[i], 0.0,
            orbit, ellipsoid, dem, targetLLH, mode.wavelength(), lookSide,
            1.0e-8, 25, 15, isce::core::HERMITE_METHOD);
        // Check
        ASSERT_EQ(stat, 1);
        ASSERT_NEAR(degrees * targetLLH[0], ref_zerodop[3*i], 1.0e-8);
        ASSERT_NEAR(degrees * targetLLH[1], ref_zerodop[3*i+1], 1.0e-8);
        ASSERT_NEAR(targetLLH[2], ref_zerodop[3*i+2], 1.0e-8);

    }
   
}

TEST_F(GeometryTest, GeoToRdr) {

    // Make a reference epoch for numerical precision
    isce::core::DateTime refEpoch(2003, 2, 25);
    orbit.updateUTCTimes(refEpoch);
    
    // Make a test LLH
    const double radians = M_PI / 180.0;
    isce::core::cartesian_t llh = {
        -115.72466801139711 * radians,
        34.65846532785868 * radians,
        1772.0
    };

    // Run geo2rdr
    double aztime, slantRange;
    int stat = isce::geometry::geo2rdr(llh, ellipsoid, orbit, skewDoppler,
        mode, aztime, slantRange, 1.0e-10, 50, 10.0);
    // Convert azimuth time to a date
    isce::core::DateTime azdate = refEpoch + aztime;

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:33.993088889");
    ASSERT_NEAR(slantRange, 830450.1859446081, 1.0e-6);

    // Run geo2rdr again with zero doppler
    isce::core::Poly2d zeroDoppler;
    stat = isce::geometry::geo2rdr(llh, ellipsoid, orbit, zeroDoppler,
        mode, aztime, slantRange, 1.0e-10, 50, 10.0);
    azdate = refEpoch + aztime;

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:34.122893704");
    ASSERT_NEAR(slantRange, 830449.6727720434, 1.0e-6);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Load test data
void loadTestData(std::vector<std::string> & aztimes, std::vector<double> & ranges,
                  std::vector<double> & heights,
                  std::vector<double> & ref_data, std::vector<double> & ref_zerodop) {

    // Load azimuth times and slant ranges
    std::ifstream ifid("input_data.txt");
    std::string line;
    while (std::getline(ifid, line)) {
        std::stringstream stream;
        std::string aztime;
        double range, h;
        stream << line;
        stream >> aztime >> range >> h;
        aztimes.push_back(aztime);
        ranges.push_back(range);
        heights.push_back(h);
    }
    ifid.close();

    // Load test data for non-zero doppler
    ifid = std::ifstream("output_data.txt");
    while (std::getline(ifid, line)) {
        std::stringstream stream;
        double lat, lon, h;
        stream << line;
        stream >> lat >> lon >> h;
        ref_data.push_back(lon);
        ref_data.push_back(lat);
        ref_data.push_back(h);
    }
    ifid.close();

    // Load test data for zero doppler
    ifid = std::ifstream("output_data_zerodop.txt");
    while (std::getline(ifid, line)) {
        std::stringstream stream;
        double lat, lon, h;
        stream << line;
        stream >> lat >> lon >> h;
        ref_zerodop.push_back(lon);
        ref_zerodop.push_back(lat);
        ref_zerodop.push_back(h);
    }
    ifid.close();

    // Check sizes
    if (aztimes.size() != (ref_data.size() / 3)) {
        std::cerr << "Incompatible data sizes" << std::endl;
        exit(1);
    }
    if (aztimes.size() != (ref_zerodop.size() / 3)) {
        std::cerr << "Incompatible data sizes" << std::endl;
        exit(1);
    }

}

// end of file
