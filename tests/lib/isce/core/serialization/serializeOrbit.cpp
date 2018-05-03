//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

// isce::core
#include <isce/core/Orbit.h>
#include <isce/core/Serialization.h>


TEST(OrbitTest, CheckArchive) {
    // Make an ellipsoid
    isce::core::Orbit orbit;

    // Open XML file
    std::ifstream xmlfid("archive.xml", std::ios::in);
    // Check if file was open successfully
    if (xmlfid.fail()) {
        std::cout << "Error: failed to open archive.xml file." << std::endl;
    }

    // Create cereal archive and load
    {
    cereal::XMLInputArchive archive(xmlfid);
    archive(cereal::make_nvp("Orbit", orbit));
    }

    // Check we have the right number of state vectors
    ASSERT_EQ(orbit.nVectors, 20);

    // Check the position of middle vector
    isce::core::StateVector sv = orbit.stateVectors[10];
    ASSERT_NEAR(sv.position()[0], -2666480.591465, 1.0e-6);
    ASSERT_NEAR(sv.position()[1], -4237357.505607, 1.0e-6);
    ASSERT_NEAR(sv.position()[2], 3958466.181821, 1.0e-6);

    // Check the velocity of middle vector
    ASSERT_NEAR(sv.velocity()[0], -198.329022, 1.0e-6);
    ASSERT_NEAR(sv.velocity()[1], 29.308432, 1.0e-6);
    ASSERT_NEAR(sv.velocity()[2], -101.540993, 1.0e-6);

    // Check date of middle vector
    ASSERT_EQ(sv.date().isoformat(), "2014-08-29T17:45:33.396381000");

}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
