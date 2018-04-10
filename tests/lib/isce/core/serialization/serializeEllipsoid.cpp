//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

#include <isce/core/Ellipsoid.h>
#include <isce/core/Serialization.h>


TEST(EllipsoidTest, CheckArchive) {
    // Make an ellipsoid
    isce::core::Ellipsoid ellipsoid;

    // Open XML file
    std::ifstream xmlfid("archive.xml", std::ios::in);
    // Check if file was open successfully
    if (xmlfid.fail()) {
        std::cout << "Error: failed to open archive.xml file." << std::endl;
    }

    // Create cereal archive and load
    {
    cereal::XMLInputArchive archive(xmlfid);
    archive(cereal::make_nvp("Ellipsoid", ellipsoid));
    }

    // Check values
    ASSERT_NEAR(ellipsoid.a(), 6378137.0, 1.0e-9);
    ASSERT_NEAR(ellipsoid.e2(), 0.0066943799, 1.0e-9);
    
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
