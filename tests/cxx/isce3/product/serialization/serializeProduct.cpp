//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>

// isce3::io
#include <isce3/io/IH5.h>

// isce3::product
#include <isce3/product/Product.h>
#include <isce3/product/Swath.h>

TEST(ProductTest, FromHDF5) {

    // Open the file
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Instantiate and load a product
    isce3::product::Product product(file);

    // Get the swath
    const isce3::product::Swath & swath = product.swath('A');

    // Check its values
    ASSERT_DOUBLE_EQ(swath.slantRange()[0], 826988.6900674499);
    ASSERT_DOUBLE_EQ(swath.zeroDopplerTime()[0], 237330.843491759);
    ASSERT_DOUBLE_EQ(swath.acquiredCenterFrequency(), 5.331004416e9);
    ASSERT_DOUBLE_EQ(swath.processedCenterFrequency(), 5.331004416e9);
    ASSERT_NEAR(swath.acquiredRangeBandwidth(), 1.6e7, 0.1);
    ASSERT_NEAR(swath.processedRangeBandwidth(), 1.6e7, 0.1);
    ASSERT_DOUBLE_EQ(swath.nominalAcquisitionPRF(), 1.0/6.051745968279355e-4);
    ASSERT_DOUBLE_EQ(swath.sceneCenterGroundRangeSpacing(), 23.774273647897644);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
