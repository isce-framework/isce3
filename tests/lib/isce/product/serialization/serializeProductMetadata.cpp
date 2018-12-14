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
#include <isce/core/LUT1d.h>

// isce::io
#include <isce/io/IH5.h>

// isce::radar
#include <isce/radar/Radar.h>

// isce::product
#include <isce/product/Serialization.h>

TEST(MetadataTest, FromHDF5) {

    // Open the file
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Instantiate a Metadata object
    isce::product::Metadata meta;

    // Open metadata group
    isce::io::IGroup metaGroup = file.openGroup("/science/metadata");

    // Deserialize the Metadata
    isce::product::loadFromH5(metaGroup, meta);

    // Get the radar instrument
    isce::radar::Radar instrument = meta.instrument();

    // Check values for content Doppler
    isce::core::LUT1d<double> content = instrument.contentDoppler();
    const size_t n = content.size();
    ASSERT_NEAR(content.values()[0], 301.353069063192, 1.0e-10);
    ASSERT_NEAR(content.values()[n-1], 278.74190662325805, 1.0e-12);

    // Check values for skew Doppler
    isce::core::LUT1d<double> skew = instrument.skewDoppler();
    ASSERT_NEAR(skew.values()[0], 301.353069063192, 1.0e-10);
    ASSERT_NEAR(skew.values()[n-1], 278.74190662325805, 1.0e-12);

    // Check range coordinates
    ASSERT_NEAR(content.coords()[0], 826988.6900674499, 1.0e-6);
    ASSERT_NEAR(content.coords()[n-1], 830882.8729292531, 1.0e-6);

    // Get the Identification
    isce::product::Identification id = meta.identification();

    // Check ellipsoid values
    ASSERT_NEAR(id.ellipsoid().a(), 6378137.0, 1.0e-9);
    ASSERT_NEAR(id.ellipsoid().e2(), 0.0066943799, 1.0e-9);

    // Get the POE orbit
    isce::core::Orbit orbit = meta.orbitPOE();

    // Copy isce::core::Orbit unit test code here
    // Check we have the right number of state vectors
    ASSERT_EQ(orbit.nVectors, 11);

    // Check the position of middle vector
    ASSERT_NEAR(orbit.position[5*3+0], -2305250.945, 1.0e-6);
    ASSERT_NEAR(orbit.position[5*3+1], -5443208.984, 1.0e-6);
    ASSERT_NEAR(orbit.position[5*3+2], 4039406.416, 1.0e-6);

    // Check the velocity of middle vector
    ASSERT_NEAR(orbit.velocity[5*3+0], -3252.930393, 1.0e-6);
    ASSERT_NEAR(orbit.velocity[5*3+1], -3129.103767, 1.0e-6);
    ASSERT_NEAR(orbit.velocity[5*3+2], -6055.488170, 1.0e-6);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
