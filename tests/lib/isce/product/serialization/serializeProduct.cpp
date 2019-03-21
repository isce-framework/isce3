//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

// isce::io
#include <isce/io/IH5.h>

// isce::product
#include <isce/product/Product.h>
#include <isce/product/Swath.h>

TEST(ProductTest, FromHDF5) {

    // Open the file
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Instantiate and load a product
    isce::product::Product product(file);

    // Get the swath
    const isce::product::Swath & swath = product.swath('A');

    // Check its values
    ASSERT_NEAR(swath.slantRange()[0], 826988.6900674499, 1.0e-5);
    ASSERT_NEAR(swath.zeroDopplerTime()[0], 237330.843491759, 1.0e-5);
    ASSERT_NEAR(swath.acquiredCenterFrequency(), 5.331004416e9, 1.0);
    ASSERT_NEAR(swath.processedCenterFrequency(), 5.331004416e9, 1.0);
    ASSERT_NEAR(swath.acquiredRangeBandwidth(), 1.6e7, 0.1);
    ASSERT_NEAR(swath.processedRangeBandwidth(), 1.6e7, 0.1);
    ASSERT_NEAR(swath.nominalAcquisitionPRF(), 1.0/6.051745968279355e-4, 1.0e-3);
    ASSERT_NEAR(swath.sceneCenterGroundRangeSpacing(), 23.774273647897644, 1.0e-8);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
