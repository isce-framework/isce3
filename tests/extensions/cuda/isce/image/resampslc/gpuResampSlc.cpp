//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Liang Yu
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

// isce::cuda::image
#include "isce/cuda/image/ResampSlc.h"


// Test that we can set radar metadata and Doppler polynomial from XML
TEST(ResampSlcTest, Resamp) {

    // Open the HDF5 product
    const std::string filename = "../../../../../lib/isce/data/envisat.h5";
    std::string h5file(filename);
    isce::io::IH5File file(h5file);

    // Create product
    isce::product::Product product(file);

    // Instantiate a ResampSLC object
    isce::cuda::image::ResampSlc gpu_resamp(product);

    // The HDF5 path to the input image
    const std::string & input_data = "HDF5:\"" + filename +
        "\"://science/LSAR/SLC/swaths/frequencyA/HH";
    
    // Perform gpu_resampling with default lines per tile
    gpu_resamp.resamp(input_data, "warped_1000.slc",
                      "../../../../../lib/isce/data/offsets/range.off",
                      "../../../../../lib/isce/data/offsets/azimuth.off");
    
    // Set lines per tile to be a weird multiple of the number of output lines
    gpu_resamp.linesPerTile(249);
    // Re-run gpu_resamp
    gpu_resamp.resamp(input_data, "warped.slc",
                      "../../../../../lib/isce/data/offsets/range.off",
                      "../../../../../lib/isce/data/offsets/azimuth.off");
}

// Compute sum of difference between reference image and warped image
TEST(ResampSlcTest, Validate) {
    // Open SLC rasters 
    isce::io::Raster refSlc("../../../../../lib/isce/data/warped_envisat.slc.vrt");
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
