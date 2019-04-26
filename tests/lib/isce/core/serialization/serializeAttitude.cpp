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
#include <isce/core/EulerAngles.h>
#include <isce/core/Serialization.h>

#include <isce/io/IH5.h>


TEST(AttitudeTest, CheckArchive) {
    // Make an attitude
    isce::core::EulerAngles euler;

    // Open the HDF5 product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Open group containing attitude
    isce::io::IGroup group = file.openGroup("/science/LSAR/SLC/metadata/attitude");

    // Deserialize the attitude
    isce::core::DateTime epoch;
    isce::core::loadFromH5(group, euler);

    // Check we have the right number of state vectors
    ASSERT_EQ(euler.nVectors(), 11);

    // Check the values of the attitude angles
    const double rad = M_PI / 180.0;
    ASSERT_NEAR(euler.yaw()[5], rad*5.0, 1.0e-10);
    ASSERT_NEAR(euler.pitch()[5], rad*5.0, 1.0e-10);
    ASSERT_NEAR(euler.roll()[5], rad*5.0, 1.0e-10);
    
    // Check date of middle vector
    isce::core::DateTime dtime = euler.refEpoch() + euler.time()[5];
    ASSERT_EQ(dtime.isoformat(), "2003-02-26T17:55:28.000000000");

}

TEST(AttitudeTest, CheckWrite) {
    // Make an attitude
    isce::core::EulerAngles euler;

    // Load orbit data
    {
    // Open the HDF5 product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Open group containing attitude
    isce::io::IGroup group = file.openGroup("/science/LSAR/SLC/metadata/attitude");

    // Deserialize the attitude
    isce::core::loadFromH5(group, euler);
    }

    // Write attitude data
    {
    // Create a dummy hdf5 file
    std::string dummyfile("dummy.h5");
    isce::io::IH5File dummy(dummyfile, 'x');

    // Write orbit to dataset
    isce::io::IGroup group = dummy.createGroup("attitude");
    isce::core::saveToH5(group, euler);
    }

    // Load a new attitude from created file
    isce::core::EulerAngles newEuler;
    std::string h5file("dummy.h5");
    isce::io::IH5File file(h5file);
    isce::io::IGroup group = file.openGroup("attitude");
    isce::core::loadFromH5(group, newEuler);

    // Check equality
    ASSERT_EQ(euler, newEuler);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
