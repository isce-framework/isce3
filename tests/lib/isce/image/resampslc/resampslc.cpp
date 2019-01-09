//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2018
//

#include <iostream>
#include <complex>
#include <string>
#include <sstream>
#include <gtest/gtest.h>
#include <cpl_conv.h>

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/Serialization.h"

// isce::io
#include "isce/io/IH5.h"
#include "isce/io/Raster.h"

// isce::product
#include "isce/product/Product.h"

// isce::image
#include "isce/image/ResampSlc.h"


// Test that we can set radar metadata and Doppler polynomial from XML
TEST(ResampSlcTest, Resamp) {

    // Open the HDF5 product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Create product
    isce::product::Product product(file);

    // Instantiate a ResampSLC object
    isce::image::ResampSlc resamp(product);

    // Use same product as a reference
    resamp.referenceProduct(product);
    
    // Check values
    ASSERT_NEAR(resamp.imageMode().startingRange(), 826988.6900674499, 1.0e-10);
    ASSERT_NEAR(resamp.doppler().values()[0], 301.353069063192, 1.0e-8);

    // Perform resampling with default lines per tile
    resamp.resamp("warped.slc", "hh",
                  "../../data/offsets/range.off", "../../data/offsets/azimuth.off");
    
    // Set lines per tile to be a weird multiple of the number of output lines
    resamp.linesPerTile(249);
    // Re-run resamp
    resamp.resamp("warped.slc", "hh",
                  "../../data/offsets/range.off", "../../data/offsets/azimuth.off");
}

// Compute sum of difference between reference image and warped image
TEST(ResampSlcTest, Validate) {
    // Open SLC rasters 
    isce::io::Raster refSlc("../../data/warped_envisat.slc.vrt");
    isce::io::Raster testSlc("warped.slc");
    // Compute total complex error
    std::complex<float> sum(0.0, 0.0);
    size_t count = 0;
    // Avoid edges of image
    for (size_t i = 20; i < (refSlc.length() - 20); ++i) {
        for (size_t j = 20; j < (refSlc.width() - 20); ++j) {
            std::complex<float> refValue, testValue;
            refSlc.getValue(refValue, j, i);
            testSlc.getValue(testValue, j, i);
            sum += testValue - refValue;
            ++count;
        }
    }
    // Normalize by number of pixels
    double abs_error = std::abs(sum) / count;
    ASSERT_LT(abs_error, 1.0e-6);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
