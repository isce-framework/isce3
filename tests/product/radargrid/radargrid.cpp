//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

// isce::product
#include <isce/product/RadarGridParameters.h>

TEST(RadarGridTest, fromProduct) {

    // Open the file
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Instantiate and load a product
    isce::product::Product product(file);

    //Create radar grid from product
    isce::product::RadarGridParameters grid(product);

    // Check its values
    ASSERT_EQ(grid.lookSide(), -1);
    ASSERT_NEAR(grid.startingRange(), 826988.6900674499, 1.0e-5);
    ASSERT_NEAR(grid.sensingStart(), 237330.843491759, 1.0e-5);
    ASSERT_NEAR(grid.wavelength(), isce::core::SPEED_OF_LIGHT/5.331004416e9, 1.0e-5);
    ASSERT_NEAR(grid.rangePixelSpacing(), 7.803973670948287, 1.0e-7);
    ASSERT_NEAR(grid.azimuthTimeInterval(), 6.051745968279355e-4, 1.0e-7);
}

TEST(RadarGridTest, fromSwath) {

    // Open the file
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Instantiate and load a product
    isce::product::Product product(file);

    // Get the swath
    const isce::product::Swath &swath = product.swath('A');

    //Create radar grid from product
    isce::product::RadarGridParameters grid(swath, -1);

    // Check its values
    ASSERT_EQ(grid.lookSide(), -1);
    ASSERT_NEAR(grid.startingRange(), 826988.6900674499, 1.0e-5);
    ASSERT_NEAR(grid.sensingStart(), 237330.843491759, 1.0e-5);
    ASSERT_NEAR(grid.wavelength(), isce::core::SPEED_OF_LIGHT/5.331004416e9, 1.0e-5);
    ASSERT_NEAR(grid.rangePixelSpacing(), 7.803973670948287, 1.0e-7);
    ASSERT_NEAR(grid.azimuthTimeInterval(), 6.051745968279355e-4, 1.0e-7);
}

TEST(RadarGridTest, fromParametersSingeLook) {
    //Reference epoch
    isce::core::DateTime t0("2017-02-12T01:12:30.0");

    //Create radar grid from product
    isce::product::RadarGridParameters grid(1, 1, 10.0,
                                           0.06, 1729.0, 800000.,
                                           10.0, 1, 24000, 6400,
                                           t0);

    // Check its values
    ASSERT_EQ(grid.lookSide(), 1);
    ASSERT_NEAR(grid.startingRange(), 800000., 1.0e-10);
    ASSERT_NEAR(grid.sensingStart(), 10.0, 1.0e-10);
    ASSERT_NEAR(grid.wavelength(), 0.06, 1.0e-10);
    ASSERT_NEAR(grid.rangePixelSpacing(), 10.0, 1.0e-10);
    ASSERT_NEAR(grid.azimuthTimeInterval(), 1/1729.0, 1.0e-8);
    ASSERT_NEAR((grid.refEpoch()-t0).getTotalSeconds(), 0.0, 1.0e-10);
    ASSERT_EQ(grid.length(), 24000);
    ASSERT_EQ(grid.width(), 6400);
}

TEST(RadarGridTest, fromParametersMultiLook) {
    //Reference epoch
    isce::core::DateTime t0("2017-02-12T01:12:30.0");

    //Create radar grid from product
    isce::product::RadarGridParameters grid(3, 4, 10.0,
                                           0.06, 1729.0, 800000.,
                                           10.0, 1, 8000, 1600,
                                           t0);

    // Check its values
    ASSERT_EQ(grid.lookSide(), 1);
    ASSERT_EQ(grid.numberAzimuthLooks(), 3);
    ASSERT_EQ(grid.numberRangeLooks(), 4);
    ASSERT_NEAR(grid.startingRangeSingleLook(), 800000., 1.0e-10);
    ASSERT_NEAR(grid.startingRange(), 800000. + 1.5 * 10.0, 1.0e-9);
    ASSERT_NEAR(grid.sensingStartSingleLook(), 10.0, 1.0e-10);
    ASSERT_NEAR(grid.sensingStart(), 10.0 + 1.0 / 1729.0, 1.0e-9);
    ASSERT_NEAR(grid.wavelength(), 0.06, 1.0e-10);
    ASSERT_NEAR(grid.rangePixelSpacingSingleLook(), 10.0, 1.0e-10);
    ASSERT_NEAR(grid.rangePixelSpacing(), 40.0, 1.0e-9);
    ASSERT_NEAR(grid.azimuthTimeIntervalSingleLook(), 1/1729.0, 1.0e-9);
    ASSERT_NEAR(grid.azimuthTimeInterval(), 3/1729.0, 1.0e-9);
    ASSERT_NEAR((grid.refEpoch()-t0).getTotalSeconds(), 0.0, 1.0e-10);
    ASSERT_EQ(grid.length(), 8000);
    ASSERT_EQ(grid.width(), 1600);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
