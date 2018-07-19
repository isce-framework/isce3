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

#include <isce/io/IH5.h>


TEST(OrbitTest, CheckArchive) {
    // Make an orbit
    isce::core::Orbit orbit;

    // Open the HDF5 product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Deserialize the orbit
    isce::core::DateTime epoch;
    isce::core::load(file, orbit, "POE", epoch);

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

    // Check date of middle vector
    isce::core::DateTime dtime = epoch + orbit.UTCtime[5];
    ASSERT_EQ(dtime.isoformat(), "2003-02-26T17:55:28.000000000");

}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
