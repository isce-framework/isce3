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
#include <isce/orbit_wip/Orbit.h>
#include <isce/core/Serialization.h>

#include <isce/io/IH5.h>


TEST(OrbitTest, CheckArchive) {
    // Make an orbit
    isce::orbit_wip::Orbit orbit;

    // Open the HDF5 product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Open group containing orbit
    isce::io::IGroup group = file.openGroup("/science/LSAR/SLC/metadata/orbit");

    // Deserialize the orbit
    isce::core::loadFromH5(group, orbit);

    // Check we have the right number of state vectors
    ASSERT_EQ(orbit.size(), 11);

    // Check the position of middle vector
    isce::core::Vec3 pos = orbit.position()[5];
    ASSERT_NEAR(pos[0], -2305250.945, 1.0e-6);
    ASSERT_NEAR(pos[1], -5443208.984, 1.0e-6);
    ASSERT_NEAR(pos[2], 4039406.416, 1.0e-6);

    // Check the velocity of middle vector
    isce::core::Vec3 vel = orbit.velocity()[5];
    ASSERT_NEAR(vel[0], -3252.930393, 1.0e-6);
    ASSERT_NEAR(vel[1], -3129.103767, 1.0e-6);
    ASSERT_NEAR(vel[2], -6055.488170, 1.0e-6);

    // Check date of middle vector
    isce::core::DateTime dtime = orbit.refepoch() + orbit.time()[5];
    ASSERT_EQ(dtime.isoformat(), "2003-02-26T17:55:28.000000000");

}

TEST(OrbitTest, CheckWrite) {
    // Make an orbit
    isce::orbit_wip::Orbit orbit;

    // Load orbit data
    {
    // Open the HDF5 product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Open group containing orbit
    isce::io::IGroup group = file.openGroup("/science/LSAR/SLC/metadata/orbit");

    // Deserialize the orbit
    isce::core::loadFromH5(group, orbit);
    }

    // Write orbit data
    {
    // Create a dummy hdf5 file
    std::string dummyfile("dummy.h5");
    isce::io::IH5File dummy(dummyfile, 'x');

    // Write orbit to dataset
    isce::io::IGroup group = dummy.createGroup("orbit");
    isce::core::saveToH5(group, orbit);
    }

    // Load a new orbit from created file
    isce::orbit_wip::Orbit newOrb;
    std::string h5file("dummy.h5");
    isce::io::IH5File file(h5file);
    isce::io::IGroup group = file.openGroup("orbit");
    isce::core::loadFromH5(group, newOrb);

    // Check equality
    for (int i=0; i < orbit.size(); i++)
    {
        isce::core::StateVector refsv = orbit[i];
        isce::core::StateVector newsv = newOrb[i];
        ASSERT_EQ(refsv, newsv);
    }
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
