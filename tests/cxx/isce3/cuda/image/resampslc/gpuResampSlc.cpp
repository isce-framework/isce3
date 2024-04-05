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

// isce3::core
#include "isce3/core/Constants.h"
#include "isce3/core/Serialization.h"

// isce3::io
#include "isce3/io/IH5.h"
#include "isce3/io/Raster.h"

// isce3::product
#include "isce3/product/RadarGridProduct.h"

// isce3::cuda::image
#include "isce3/cuda/image/ResampSlc.h"


// Test that we can set radar metadata and Doppler polynomial from XML
TEST(ResampSlcTest, Resamp) {

    // Open the HDF5 product
    const std::string filename = TESTDATA_DIR "envisat.h5";
    std::string h5file(filename);
    isce3::io::IH5File file(h5file);

    // Create product
    isce3::product::RadarGridProduct product(file);

    // Instantiate a ResampSLC object
    isce3::cuda::image::ResampSlc gpu_resamp(product);

    // The HDF5 path to the input image
    const std::string & input_data = "HDF5:\"" + filename +
        "\"://science/LSAR/SLC/swaths/frequencyA/HH";

    // Perform resampling with default lines per tile allows resample to be run
    // in a single block.
    gpu_resamp.resamp(input_data, "warped_single_block.slc",
                      TESTDATA_DIR "offsets/range.off",
                      TESTDATA_DIR "offsets/azimuth.off");

    // Set lines per tile such that multiple blocks are processed.
    gpu_resamp.linesPerTile(249);
    // Re-run gpu_resamp
    gpu_resamp.resamp(input_data, "warped_many_blocks.slc",
                      TESTDATA_DIR "offsets/range.off",
                      TESTDATA_DIR "offsets/azimuth.off");
}

// Compute sum of difference between reference image and warped image
TEST(ResampSlcTest, Validate) {
    // Open SLC reference raster
    isce3::io::Raster refSlc(TESTDATA_DIR "warped_envisat.slc.vrt");

    // Iterate over single and multiple block outputs.
    const auto output_files = {"warped_single_block.slc",
                               "warped_many_blocks.slc"};
    for (auto output_file : output_files) {
        isce3::io::Raster testSlc(output_file);

        // Compute total complex error
        float sum{0.0};
        size_t count = 0;
        // Avoid edges of image
        for (size_t i = 20; i < (refSlc.length() - 20); ++i) {
            for (size_t j = 20; j < (refSlc.width() - 20); ++j) {
                std::complex<float> refValue, testValue;
                refSlc.getValue(refValue, j, i);
                testSlc.getValue(testValue, j, i);
                sum += std::abs(testValue - refValue);
                ++count;
            }
        }
        // Normalize by number of pixels
        double abs_error = sum / count;
        ASSERT_LT(abs_error, 4.5e-5);
    }
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
