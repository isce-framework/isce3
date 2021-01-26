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
#include <isce3/core/Attitude.h>
#include <isce3/core/EulerAngles.h>
#include <isce3/core/Quaternion.h>
#include <isce3/core/Serialization.h>

#include <isce3/io/IH5.h>


TEST(AttitudeTest, CheckArchive) {
    // Make an attitude
    isce3::core::Attitude attitude;

    // Open the HDF5 product
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Open group containing attitude
    isce3::io::IGroup group = file.openGroup("/science/LSAR/SLC/metadata/attitude");

    // Deserialize the attitude
    isce3::core::DateTime epoch;
    isce3::core::loadFromH5(group, attitude);

    // Check we have the right number of state vectors
    ASSERT_EQ(attitude.size(), 11);

    // Check the values of the attitude angles
    const auto q = attitude.quaternions()[5];
    const auto expected = isce3::core::Quaternion(5, 5, 5, 5);
    EXPECT_DOUBLE_EQ(q.w(), expected.w());
    EXPECT_DOUBLE_EQ(q.x(), expected.x());
    EXPECT_DOUBLE_EQ(q.y(), expected.y());
    EXPECT_DOUBLE_EQ(q.z(), expected.z());

    // Check date of middle vector
    isce3::core::DateTime dtime = attitude.referenceEpoch() + attitude.time()[5];
    ASSERT_EQ(dtime.isoformat(), "2003-02-26T17:55:28.000000000");

}

TEST(AttitudeTest, CheckWrite) {
    // Make an attitude
    isce3::core::Attitude attitude;

    // Load orbit data
    {
    // Open the HDF5 product
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Open group containing attitude
    isce3::io::IGroup group = file.openGroup("/science/LSAR/SLC/metadata/attitude");

    // Deserialize the attitude
    isce3::core::loadFromH5(group, attitude);
    }

    // Write attitude data
    {
    // Create a dummy hdf5 file
    std::string dummyfile("dummy.h5");
    isce3::io::IH5File dummy(dummyfile, 'x');

    // Write orbit to dataset
    isce3::io::IGroup group = dummy.createGroup("attitude");
    isce3::core::saveToH5(group, attitude);
    }

    // Load a new attitude from created file
    isce3::core::Attitude newAttitude;
    std::string h5file("dummy.h5");
    isce3::io::IH5File file(h5file);
    isce3::io::IGroup group = file.openGroup("attitude");
    isce3::core::loadFromH5(group, newAttitude);

    // Check equality
    ASSERT_EQ(attitude.size(), newAttitude.size());
    ASSERT_EQ(attitude.referenceEpoch(), newAttitude.referenceEpoch());
    for (int i = 0; i < attitude.size(); ++i) {
        ASSERT_DOUBLE_EQ(attitude.time()[i], newAttitude.time()[i]);
        auto expected = attitude.quaternions()[i];
        auto got = newAttitude.quaternions()[i];
        ASSERT_DOUBLE_EQ(expected.w(), got.w());
        ASSERT_DOUBLE_EQ(expected.x(), got.x());
        ASSERT_DOUBLE_EQ(expected.y(), got.y());
        ASSERT_DOUBLE_EQ(expected.z(), got.z());
    }
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
