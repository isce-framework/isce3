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
#include <isce/core/DateTime.h>

// isce::io
#include <isce/io/IH5.h>

// isce::product
#include <isce/product/Serialization.h>

TEST(ImageModeTest, FromHDF5) {

    // Open the file
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Instantiate an ImageMode object
    isce::product::ImageMode mode;

    // Open group for image mode
    isce::io::IGroup modeGroup = file.openGroup("/science/complex_imagery");
    
    // Deserialize the primary_mode
    isce::product::loadFromH5(modeGroup, mode, "aux");

    // Check values
    ASSERT_NEAR(mode.rangePixelSpacing(), 7.803973670948287, 1.0e-10);
    ASSERT_NEAR(mode.startingRange(), 826988.6900674499, 1.0e-10);
    ASSERT_EQ(mode.startAzTime().isoformat(), "2003-02-26T17:55:30.843491759");
    ASSERT_NEAR(mode.prf(), 1652.415691672402, 1.0e-10);
    ASSERT_NEAR(mode.wavelength(), 0.05623564240544047, 1.0e-10);
    ASSERT_NEAR(mode.rangeBandwidth(), 1.6e7, 0.1);

}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
