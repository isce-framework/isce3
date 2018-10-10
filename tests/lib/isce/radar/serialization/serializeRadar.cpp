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
#include <isce/core/Poly2d.h>

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
    isce::core::Poly2d content = instrument.contentDoppler();
    ASSERT_NEAR(content.coeffs[0], 301.353069063192, 1.0e-10);
    ASSERT_NEAR(content.coeffs[1], -0.04633312447837376, 1.0e-10);
    ASSERT_NEAR(content.coeffs[2], 2.044436266418998e-6, 1.0e-12);

    // Check values for skew Doppler
    isce::core::Poly2d skew = instrument.skewDoppler();
    ASSERT_NEAR(content.coeffs[0], 301.353069063192, 1.0e-10);
    ASSERT_NEAR(content.coeffs[1], -0.04633312447837376, 1.0e-10);
    ASSERT_NEAR(content.coeffs[2], 2.044436266418998e-6, 1.0e-12);
    
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
