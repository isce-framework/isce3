//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <gtest/gtest.h>

#include "isce/core/Doppler.h"

isce::core::Orbit loadOrbitData();

struct DopplerTest : public ::testing::Test {

    // Some useful typedefs
    typedef isce::core::Ellipsoid Ellipsoid;
    typedef isce::core::EulerAngles EulerAngles;
    typedef isce::core::Orbit Orbit;

    double yaw, pitch, roll, t_az, slantRange, wvl, fd, tol;
    EulerAngles attitude;
    Ellipsoid ellipsoid;
    std::vector<double> ranges, wvls, fd_ref;

    protected:

        DopplerTest() {

            // Set the attitude angles
            yaw = 0.1;
            pitch = 0.05;
            roll = -0.1;

            // Set epoch
            t_az = 11900.0;

            // Set the attitude
            attitude = EulerAngles(yaw, pitch, roll);
    
            // Make ellipsoid
            ellipsoid = Ellipsoid(isce::core::EarthSemiMajorAxis,
                isce::core::EarthEccentricitySquared);

            // Set the slant range pixels to test
            ranges = {847417.0, 847467.0, 847517.0, 847567.0};
            // Set the wavelengths to test
            wvls = {0.21, 0.22, 0.23, 0.24};
            // Set the reference Doppler values
            fd_ref = {-3375.49106567, -3222.95549713, -3083.68365585, -2956.01757605};

        }

        ~DopplerTest() {
            ranges.clear();
            wvls.clear();
            fd_ref.clear(); 
        }
};

// Check for successful construction of a doppler object
TEST_F(DopplerTest, CheckConstruction) {
    Orbit orbit = loadOrbitData();
    isce::core::Doppler doppler = isce::core::Doppler(orbit, &attitude, ellipsoid, t_az);
} 

// Check computation of Doppler centroid
TEST_F(DopplerTest, CheckCentroid) {
    Orbit orbit = loadOrbitData();
    isce::core::Doppler doppler = isce::core::Doppler(orbit, &attitude, ellipsoid, t_az);
    for (size_t i = 0; i < fd_ref.size(); ++i) {
        double fd = doppler.centroid(ranges[i], wvls[i], "inertial");
        ASSERT_NEAR(fd, fd_ref[i], 1.0e-5);
    }
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

isce::core::Orbit loadOrbitData() {

    // Instantiate an empty orbit
    isce::core::Orbit orbit;

    // Open file for reading
    std::ifstream fid("orbit_data.txt");
    // Check if file open was successful
    if (fid.fail()) {
        std::cout << "Error: Failed to open orbit file for Doppler test." << std::endl;
    }

    // Loop over state vectors 
    while (fid) {
        std::string str;
        std::stringstream stream;
        std::getline(fid, str);
        if (str.length() < 1)
            break;
        stream << str;
        double t;
        stream >> t;
        std::stringstream().swap(stream);

        std::getline(fid, str);
        stream << str;
        double p0, p1, p2;
        stream >> p0 >> p1 >> p2;
        std::stringstream().swap(stream);

        std::getline(fid, str);
        stream << str;
        double v0, v1, v2;
        stream >> v0 >> v1 >> v2;
        std::stringstream().swap(stream);

        // Add data to orbit
        orbit.UTCtime.push_back(t);
        orbit.position.push_back(p0);
        orbit.position.push_back(p1);
        orbit.position.push_back(p2);
        orbit.velocity.push_back(v0);
        orbit.velocity.push_back(v1);
        orbit.velocity.push_back(v2);
    }

    // Close the file
    fid.close();

    // Update other orbit attributes
    orbit.nVectors = orbit.UTCtime.size();

    return orbit;
}

// end of file
