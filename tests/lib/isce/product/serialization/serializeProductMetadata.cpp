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
#include <isce/core/LUT2d.h>

// isce::io
#include <isce/io/IH5.h>

// isce::product
#include <isce/product/Serialization.h>

TEST(MetadataTest, FromHDF5) {

    // Open the file
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Instantiate a Metadata object
    isce::product::Metadata meta;

    // Open metadata group
    isce::io::IGroup metaGroup = file.openGroup("/science/LSAR/SLC/metadata");

    // Deserialize the Metadata
    isce::product::loadFromH5(metaGroup, meta);

    // Get the orbit
    const isce::core::Orbit & orbit = meta.orbit();

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

    // Get the attitude
    const isce::core::EulerAngles & euler = meta.attitude();

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

    // Check the ProcessingInformation
    const isce::product::ProcessingInformation & proc = meta.procInfo();
    ASSERT_NEAR(proc.slantRange()[0], 826000.0, 1.0e-10);
    ASSERT_NEAR(proc.zeroDopplerTime()[0], 237321.0, 1.0e-10);

    // Check effective velocity LUT    
    const isce::core::LUT2d<double> & veff = proc.effectiveVelocity();
    ASSERT_EQ(veff.width(), 240);
    ASSERT_EQ(veff.length(), 80);
    ASSERT_NEAR(veff.xStart(), 826000.0, 1.0e-10);
    ASSERT_NEAR(veff.yStart(), 237321.0, 1.0e-10);
    ASSERT_NEAR(veff.xSpacing(), 25.0, 1.0e-10);
    ASSERT_NEAR(veff.ySpacing(), 0.25, 1.0e-10);
    ASSERT_NEAR(veff.data()(10, 35), 7112.3142687909785, 1.0e-6);

    // Check Doppler centroid
    const isce::core::LUT2d<double> & dopp = proc.dopplerCentroid('A');
    ASSERT_NEAR(dopp.data()(10, 35), 302.0284944801832, 1.0e-6);

    // Check azimuth FM rate
    const isce::core::LUT2d<double> & fmrate = proc.azimuthFMRate('A');
    ASSERT_NEAR(fmrate.data()(10, 35), 2175.7067054603435, 1.0e-6);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
