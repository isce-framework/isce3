//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

#include <isce/core/Poly2d.h>
#include <isce/core/Serialization.h>

#include <isce/io/IH5.h>

TEST(DopplerTest, CheckArchive) {
    // Make a Poly2D for Dopple representation
    isce::core::Poly2d doppler;

    // Open HDF5 file
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Open group containing Doppler
    isce::io::IGroup group = file.openGroup("/science/metadata/instrument_data/doppler_centroid");

    // Deserialize the Doppler
    isce::core::loadFromH5(group, doppler, "data_dcpolynomial");
    
    // Check values
    ASSERT_NEAR(doppler.coeffs[0], 301.353069063192, 1.0e-10);
    ASSERT_NEAR(doppler.coeffs[1], -0.04633312447837376, 1.0e-10);
    ASSERT_NEAR(doppler.coeffs[2], 2.044436266418998e-6, 1.0e-12);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
