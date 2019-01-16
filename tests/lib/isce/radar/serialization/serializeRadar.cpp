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
#include <isce/radar/Serialization.h>

TEST(RadarTest, CheckArchive) {

    // Make a Radar object
    isce::radar::Radar instrument;

    // Open the file
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Open group containing instrument data
    isce::io::IGroup group = file.openGroup("/science/metadata/instrument_data");

    // Deserialize the radar instrument
    isce::radar::loadFromH5(group, instrument);

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
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
