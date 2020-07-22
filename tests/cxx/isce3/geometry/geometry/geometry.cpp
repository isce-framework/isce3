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

// isce3::io
#include <isce3/io/IH5.h>

// isce3::core
#include <isce3/core/Constants.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Serialization.h>
#include <isce3/core/TimeDelta.h>

// isce3::product
#include <isce3/product/Product.h>

// isce3::geometry
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/geometry.h>

// Declaration for utility function to read test data
void loadTestData(std::vector<std::string> & aztimes, std::vector<double> & ranges,
                  std::vector<double> & heights,
                  std::vector<double> & ref_data, std::vector<double> & ref_zerodop);

struct GeometryTest : public ::testing::Test {

    // isce3::core objects
    isce3::core::Ellipsoid ellipsoid;
    isce3::core::LUT2d<double> doppler;
    isce3::core::Orbit orbit;

    // isce3::product objects
    isce3::product::ProcessingInformation proc;
    isce3::product::Swath swath;

    isce3::core::LookSide lookSide;

    // Constructor
    protected:
        GeometryTest() {
            // Open the HDF5 product
            std::string h5file(TESTDATA_DIR "envisat.h5");
            isce3::io::IH5File file(h5file);

            // Instantiate a Product
            isce3::product::Product product(file);

            // Extract core and product objects
            orbit = product.metadata().orbit();
            proc = product.metadata().procInfo();
            swath = product.swath('A');
            doppler = proc.dopplerCentroid('A');
            lookSide = product.lookSide();
            ellipsoid.a(isce3::core::EarthSemiMajorAxis);
            ellipsoid.e2(isce3::core::EarthEccentricitySquared);

            // For this test, use biquintic interpolation for Doppler LUT
            doppler.interpMethod(isce3::core::BIQUINTIC_METHOD);
        }
};

TEST_F(GeometryTest, RdrToGeoWithOrbit) {

    // Load test data
    std::vector<std::string> aztimes;
    std::vector<double> ranges, heights, ref_data, ref_zerodop;
    loadTestData(aztimes, ranges, heights, ref_data, ref_zerodop);

    // Loop over test data
    const double degrees = 180.0 / M_PI;
    for (size_t i = 0; i < aztimes.size(); ++i) {

        // Make azimuth time in seconds
        isce3::core::DateTime azDate = aztimes[i];
        const double azTime = (azDate - orbit.referenceEpoch()).getTotalSeconds();

        // Evaluate Doppler
        const double dopval = doppler.eval(azTime, ranges[i]);

        // Make constant DEM interpolator set to input height
        isce3::geometry::DEMInterpolator dem(heights[i]);

        // Initialize guess
        isce3::core::cartesian_t targetLLH = {0.0, 0.0, heights[i]};

        // Run rdr2geo
        int stat = isce3::geometry::rdr2geo(azTime, ranges[i], dopval,
            orbit, ellipsoid, dem, targetLLH, swath.processedWavelength(), lookSide,
            1.0e-8, 25, 15);

        // Check
        ASSERT_EQ(stat, 1);
        ASSERT_NEAR(degrees * targetLLH[0], ref_data[3*i], 1.0e-8);
        ASSERT_NEAR(degrees * targetLLH[1], ref_data[3*i+1], 1.0e-8);
        ASSERT_NEAR(targetLLH[2], ref_data[3*i+2], 1.0e-8);

        // Run again with zero doppler
        stat = isce3::geometry::rdr2geo(azTime, ranges[i], 0.0,
            orbit, ellipsoid, dem, targetLLH, swath.processedWavelength(), lookSide,
            1.0e-8, 25, 15);
        // Check
        ASSERT_EQ(stat, 1);
        ASSERT_NEAR(degrees * targetLLH[0], ref_zerodop[3*i], 1.0e-8);
        ASSERT_NEAR(degrees * targetLLH[1], ref_zerodop[3*i+1], 1.0e-8);
        ASSERT_NEAR(targetLLH[2], ref_zerodop[3*i+2], 1.0e-8);

    }

}

TEST_F(GeometryTest, GeoToRdr) {

    // Make a test LLH
    const double radians = M_PI / 180.0;
    isce3::core::cartesian_t llh = {
        -115.72466801139711 * radians,
        34.65846532785868 * radians,
        1772.0
    };

    // Run geo2rdr
    double aztime, slantRange;
    int stat = isce3::geometry::geo2rdr(llh, ellipsoid, orbit, doppler,
        aztime, slantRange, swath.processedWavelength(), lookSide,
        1.0e-10, 50, 10.0);
    // Convert azimuth time to a date
    isce3::core::DateTime azdate = orbit.referenceEpoch() + aztime;

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:33.993088889");
    ASSERT_NEAR(slantRange, 830450.1859446081, 1.0e-6);

    // Run geo2rdr again with zero doppler
    isce3::core::LUT2d<double> zeroDoppler;
    stat = isce3::geometry::geo2rdr(llh, ellipsoid, orbit, zeroDoppler,
        aztime, slantRange, swath.processedWavelength(), lookSide,
        1.0e-10, 50, 10.0);
    azdate = orbit.referenceEpoch() + aztime;

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
