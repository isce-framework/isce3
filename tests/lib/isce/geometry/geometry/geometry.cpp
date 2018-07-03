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
#include "isce/core/LinAlg.h"
#include "isce/core/Orbit.h"
#include "isce/core/Peg.h"
#include "isce/core/Pegtrans.h"
#include "isce/core/Poly2d.h"
#include "isce/core/Metadata.h"
#include "isce/core/Serialization.h"

// isce::geometry
#include "isce/geometry/geometry.h"

// Declaration for utility function to read test data
void loadTestData(std::vector<std::string> & aztimes, std::vector<double> & ranges,
                  std::vector<double> & heights,
                  std::vector<double> & ref_data, std::vector<double> & ref_zerodop);

struct GeometryTest : public ::testing::Test {

    // isce::core objects
    isce::core::Ellipsoid ellipsoid;
    isce::core::Poly2d skewDoppler;
    isce::core::Metadata meta;
    isce::core::Orbit orbit;

    // Constructor
    protected:
        GeometryTest() {
            // Open the HDF5 product
            std::string h5file("../../data/envisat.h5");
            isce::io::IH5File file(h5file);

            // Deserialization
            isce::core::load(file, ellipsoid);
            isce::core::load(file, orbit, "POE");
            isce::core::load(file, skewDoppler, "skew");
            isce::core::load(file, meta, "primary");
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

        // Make azimuth date
        const isce::core::DateTime azDate(aztimes[i]); 
        const double azTime = azDate.secondsSinceEpoch();

        // Evaluate Doppler
        const double rbin = (ranges[i] - meta.rangeFirstSample) / meta.slantRangePixelSpacing;
        const double doppler = skewDoppler.eval(0, rbin);

        // Make constant DEM interpolator set to input height
        isce::geometry::DEMInterpolator dem(heights[i]);

        // Initialize guess
        isce::core::cartesian_t targetLLH = {0.0, 0.0, dem.interpolateLonLat(0.0, 0.0)};

        // Run rdr2geo
        int stat = isce::geometry::rdr2geo(azTime, ranges[i], doppler,
            orbit, ellipsoid, dem, targetLLH, meta.radarWavelength, meta.lookSide,
            1.0e-8, 25, 15, isce::core::HERMITE_METHOD);
        // Check
        ASSERT_EQ(stat, 1);
        ASSERT_NEAR(degrees * targetLLH[0], ref_data[3*i], 1.0e-8);
        ASSERT_NEAR(degrees * targetLLH[1], ref_data[3*i+1], 1.0e-8);
        ASSERT_NEAR(targetLLH[2], ref_data[3*i+2], 1.0e-8);

        // Run again with zero doppler
        stat = isce::geometry::rdr2geo(azTime, ranges[i], 0.0,
            orbit, ellipsoid, dem, targetLLH, meta.radarWavelength, meta.lookSide,
            1.0e-8, 25, 15, isce::core::HERMITE_METHOD);
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
    isce::core::cartesian_t llh = {-115.6*radians, 35.10*radians, 55.0};

    // Run geo2rdr
    double aztime, slantRange;
    int stat = isce::geometry::geo2rdr(llh, ellipsoid, orbit, skewDoppler,
        meta, aztime, slantRange, 1.0e-8, 50, 1.0e-8);
    // Convert azimuth time to a date
    isce::core::DateTime azdate;
    azdate.secondsSinceEpoch(aztime);

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:26.487007976");
    ASSERT_NEAR(slantRange, 831834.3551143121, 1.0e-6);

    // Run geo2rdr again with zero doppler
    isce::core::Poly2d zeroDoppler;
    stat = isce::geometry::geo2rdr(llh, ellipsoid, orbit, zeroDoppler,
        meta, aztime, slantRange, 1.0e-8, 50, 1.0e-8);
    azdate.secondsSinceEpoch(aztime);

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:26.613449931");
    ASSERT_NEAR(slantRange, 831833.869159697, 1.0e-6);
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
