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
#include <isce/core/LookSide.h>
#include <isce/product/RadarGridParameters.h>

using isce::core::LookSide;

TEST(RadarGridTest, fromProduct) {

    // Open the file
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce::io::IH5File file(h5file);

    // Instantiate and load a product
    isce::product::Product product(file);

    //Create radar grid from product
    isce::product::RadarGridParameters grid(product);

    // Check its values
    ASSERT_EQ(grid.lookSide(), LookSide::Right);
    ASSERT_NEAR(grid.startingRange(), 826988.6900674499, 1.0e-5);
    ASSERT_NEAR(grid.sensingStart(), 237330.843491759, 1.0e-5);
    ASSERT_NEAR(grid.wavelength(), isce::core::SPEED_OF_LIGHT/5.331004416e9, 1.0e-5);
    ASSERT_NEAR(grid.rangePixelSpacing(), 7.803973670948287, 1.0e-7);
    ASSERT_NEAR(grid.azimuthTimeInterval(), 6.051745968279355e-4, 1.0e-7);
}

TEST(RadarGridTest, fromSwath) {

    // Open the file
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce::io::IH5File file(h5file);

    // Instantiate and load a product
    isce::product::Product product(file);

    // Get the swath
    const isce::product::Swath &swath = product.swath('A');

    //Create radar grid from product
    isce::product::RadarGridParameters grid(swath, LookSide::Right);

    // Check its values
    ASSERT_EQ(grid.lookSide(), LookSide::Right);
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
    isce::product::RadarGridParameters grid(10.0,
                                           0.06, 1729.0, 800000.,
                                           10.0, LookSide::Left, 24000, 6400,
                                           t0);

    // Check its values
    ASSERT_EQ(grid.lookSide(), LookSide::Left);
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
    isce::product::RadarGridParameters grid(10.0,
                                           0.06, 1729.0, 800000.,
                                           10.0, LookSide::Left, 8000, 1600,
                                           t0);

    //Multilook
    isce::product::RadarGridParameters mlgrid = grid.multilook(3, 4);

    // Check its values
    ASSERT_EQ(mlgrid.lookSide(), LookSide::Left);
    ASSERT_NEAR(mlgrid.startingRange(), 800000. + 1.5 * 10.0, 1.0e-9);
    ASSERT_NEAR(mlgrid.sensingStart(), 10.0 + 1.0 / 1729.0, 1.0e-9);
    ASSERT_NEAR(mlgrid.wavelength(), 0.06, 1.0e-10);
    ASSERT_NEAR(mlgrid.rangePixelSpacing(), 40.0, 1.0e-9);
    ASSERT_NEAR(mlgrid.azimuthTimeInterval(), 3/1729.0, 1.0e-9);
    ASSERT_NEAR((mlgrid.refEpoch()-t0).getTotalSeconds(), 0.0, 1.0e-10);
    ASSERT_EQ(mlgrid.length(), 8000/3);
    ASSERT_EQ(mlgrid.width(), 1600/4);
}


TEST(RadarGridTest, fromParametersCrop) {
    //Reference epoch
    isce::core::DateTime t0("2017-02-12T01:12:30.0");

    //Create radar grid from product
    isce::product::RadarGridParameters grid(10.0,
                                           0.06, 1729.0, 800000.,
                                           10.0, LookSide::Left, 8000, 1600,
                                           t0);

    //Crop
    isce::product::RadarGridParameters mlgrid = grid.offsetAndResize(400, 500, 2000, 800);

    // Check its values
    ASSERT_EQ(mlgrid.lookSide(), LookSide::Left);
    ASSERT_NEAR(mlgrid.startingRange(), grid.slantRange(500), 1.0e-9);
    ASSERT_NEAR(mlgrid.sensingStart(), grid.sensingTime(400), 1.0e-9);
    ASSERT_EQ(mlgrid.wavelength(), grid.wavelength());
    ASSERT_EQ(mlgrid.rangePixelSpacing(), grid.rangePixelSpacing());
    ASSERT_EQ(mlgrid.azimuthTimeInterval(), grid.azimuthTimeInterval());
    ASSERT_NEAR((mlgrid.refEpoch()-grid.refEpoch()).getTotalSeconds(), 0.0, 1.0e-10);
    ASSERT_EQ(mlgrid.length(), 2000);
    ASSERT_EQ(mlgrid.width(), 800);
}


TEST(RadarGridTest, singleLook)
{

    //Reference epoch
    isce::core::DateTime t0("2017-02-12T01:12:30.0");

    //Create radar grid from product
    isce::product::RadarGridParameters grid(10.0,
                                           0.06, 1729.0, 800000.,
                                           10.0, LookSide::Left, 8000, 1600,
                                           t0);

    //Multilook
    isce::product::RadarGridParameters mlgrid = grid.multilook(1, 1);

    ASSERT_EQ( grid.lookSide(), mlgrid.lookSide());
    ASSERT_NEAR(grid.startingRange(), mlgrid.startingRange(), 1.0e-11);
    ASSERT_NEAR(grid.sensingStart(), mlgrid.sensingStart(), 1.0e-11);
    ASSERT_EQ(grid.wavelength(), mlgrid.wavelength());
    ASSERT_EQ(grid.rangePixelSpacing(), mlgrid.rangePixelSpacing());
    ASSERT_EQ(grid.azimuthTimeInterval(), mlgrid.azimuthTimeInterval());
    ASSERT_NEAR((mlgrid.refEpoch()-grid.refEpoch()).getTotalSeconds(), 0.0, 1.0e-11);
    ASSERT_EQ(grid.length(), mlgrid.length());
    ASSERT_EQ(grid.width(), mlgrid.width());

}

TEST(RadarGridTest, cropSame)
{

    //Reference epoch
    isce::core::DateTime t0("2017-02-12T01:12:30.0");

    //Create radar grid from product
    isce::product::RadarGridParameters grid(10.0,
                                           0.06, 1729.0, 800000.,
                                           10.0, LookSide::Left, 8000, 1600,
                                           t0);

    //Multilook
    isce::product::RadarGridParameters mlgrid = grid.offsetAndResize(0,0, grid.length(), grid.width());

    ASSERT_EQ( grid.lookSide(), mlgrid.lookSide());
    ASSERT_NEAR(grid.startingRange(), mlgrid.startingRange(), 1.0e-11);
    ASSERT_NEAR(grid.sensingStart(), mlgrid.sensingStart(), 1.0e-11);
    ASSERT_EQ(grid.wavelength(), mlgrid.wavelength());
    ASSERT_EQ(grid.rangePixelSpacing(), mlgrid.rangePixelSpacing());
    ASSERT_EQ(grid.azimuthTimeInterval(), mlgrid.azimuthTimeInterval());
    ASSERT_NEAR((mlgrid.refEpoch()-grid.refEpoch()).getTotalSeconds(), 0.0, 1.0e-11);
    ASSERT_EQ(grid.length(), mlgrid.length());
    ASSERT_EQ(grid.width(), mlgrid.width());

}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
