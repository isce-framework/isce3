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
#include <isce3/core/LUT2d.h>
#include <isce3/core/Quaternion.h>

// isce3::io
#include <isce3/io/IH5.h>

// isce3::product
#include <isce3/product/Serialization.h>

TEST(MetadataTest, FromHDF5) {

    // Open the file
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Instantiate a Metadata object
    isce3::product::Metadata meta;

    // Open metadata group
    isce3::io::IGroup metaGroup = file.openGroup("/science/LSAR/SLC/metadata");

    // Deserialize the Metadata
    isce3::product::loadFromH5(metaGroup, meta);

    // Get the orbit
    const isce3::core::Orbit & orbit = meta.orbit();

    // Copy isce3::core::Orbit unit test code here
    // Check we have the right number of state vectors
    ASSERT_EQ(orbit.size(), 11);

    // Check the position of middle vector
    ASSERT_NEAR(orbit.position(5)[0], -2305250.945, 1.0e-6);
    ASSERT_NEAR(orbit.position(5)[1], -5443208.984, 1.0e-6);
    ASSERT_NEAR(orbit.position(5)[2], 4039406.416, 1.0e-6);

    // Check the velocity of middle vector
    ASSERT_NEAR(orbit.velocity(5)[0], -3252.930393, 1.0e-6);
    ASSERT_NEAR(orbit.velocity(5)[1], -3129.103767, 1.0e-6);
    ASSERT_NEAR(orbit.velocity(5)[2], -6055.488170, 1.0e-6);

    // Get the attitude
    const auto attitude = meta.attitude();

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

    // Check the ProcessingInformation
    const isce3::product::ProcessingInformation & proc = meta.procInfo();

    // Check effective velocity LUT
    const isce3::core::LUT2d<double> & veff = proc.effectiveVelocity();
    ASSERT_EQ(veff.width(), 240);
    ASSERT_EQ(veff.length(), 80);
    ASSERT_NEAR(veff.xStart(), 826000.0, 1.0e-10);
    ASSERT_NEAR(veff.yStart(), 237321.0, 1.0e-10);
    ASSERT_NEAR(veff.xSpacing(), 25.0, 1.0e-10);
    ASSERT_NEAR(veff.ySpacing(), 0.25, 1.0e-10);
    ASSERT_NEAR(veff.data()(10, 35), 7112.3142687909785, 1.0e-6);

    // Check Doppler centroid
    const isce3::core::LUT2d<double> & dopp = proc.dopplerCentroid('A');
    ASSERT_NEAR(dopp.data()(10, 35), 302.0284944801832, 1.0e-6);

    // Check azimuth FM rate
    const isce3::core::LUT2d<double> & fmrate = proc.azimuthFMRate('A');
    ASSERT_NEAR(fmrate.data()(10, 35), 2175.7067054603435, 1.0e-6);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
