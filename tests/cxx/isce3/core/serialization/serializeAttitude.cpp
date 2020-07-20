//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

// isce3::core
#include <isce3/core/EulerAngles.h>
#include <isce3/core/Serialization.h>

#include <isce3/io/IH5.h>


TEST(AttitudeTest, CheckArchive) {
    // Make an attitude
    isce3::core::EulerAngles euler;

    // Open the HDF5 product
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Open group containing attitude
    isce3::io::IGroup group = file.openGroup("/science/LSAR/SLC/metadata/attitude");

    // Deserialize the attitude
    isce3::core::DateTime epoch;
    isce3::core::loadFromH5(group, euler);

    // Check we have the right number of state vectors
    ASSERT_EQ(euler.nVectors(), 11);

    // Check the values of the attitude angles
    const double rad = M_PI / 180.0;
    ASSERT_NEAR(euler.yaw()[5], rad*5.0, 1.0e-10);
    ASSERT_NEAR(euler.pitch()[5], rad*5.0, 1.0e-10);
    ASSERT_NEAR(euler.roll()[5], rad*5.0, 1.0e-10);
    
    // Check date of middle vector
    isce3::core::DateTime dtime = euler.refEpoch() + euler.time()[5];
    ASSERT_EQ(dtime.isoformat(), "2003-02-26T17:55:28.000000000");

}

TEST(AttitudeTest, CheckWrite) {
    // Make an attitude
    isce3::core::EulerAngles euler;

    // Load orbit data
    {
    // Open the HDF5 product
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Open group containing attitude
    isce3::io::IGroup group = file.openGroup("/science/LSAR/SLC/metadata/attitude");

    // Deserialize the attitude
    isce3::core::loadFromH5(group, euler);
    }

    // Write attitude data
    {
    // Create a dummy hdf5 file
    std::string dummyfile("dummy.h5");
    isce3::io::IH5File dummy(dummyfile, 'x');

    // Write orbit to dataset
    isce3::io::IGroup group = dummy.createGroup("attitude");
    isce3::core::saveToH5(group, euler);
    }

    // Load a new attitude from created file
    isce3::core::EulerAngles newEuler;
    std::string h5file("dummy.h5");
    isce3::io::IH5File file(h5file);
    isce3::io::IGroup group = file.openGroup("attitude");
    isce3::core::loadFromH5(group, newEuler);

    // Check equality
    ASSERT_EQ(euler, newEuler);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
