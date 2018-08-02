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
#include "isce/core/ResampSlc.h"
#include "isce/core/Serialization.h"

// isce::io
#include "isce/io/Raster.h"

// Declaration for utility function to read metadata stream from VRT
std::stringstream streamFromVRT(const char * filename, int bandNum=1);

struct ResampSlcTest : public ::testing::Test {
    // Typedefs
    typedef isce::core::ResampSlc ResampSlc;
    // Data
    std::stringstream metastream;
    ResampSlc resamp;
    // Constructor
    protected:
        ResampSlcTest() {
            // Load metadata stream
            metastream = streamFromVRT("../../data/envisat.slc.vrt");
        }
};

// Test that we can set radar metadata and Doppler polynomial from XML
TEST_F(ResampSlcTest, Resamp) {
    // Deserialize local metadata 
    isce::core::Metadata meta, refMeta;
    isce::core::Poly2d doppler;
    {
    cereal::XMLInputArchive archive(metastream);
    archive(cereal::make_nvp("ContentDoppler", doppler),
            cereal::make_nvp("Radar", meta),
            cereal::make_nvp("Radar", refMeta));
    }
    // Set resamp metadata and Doppler
    resamp.metadata(meta);
    resamp.refMetadata(refMeta);
    resamp.doppler(doppler);
    // Check values
    ASSERT_NEAR(resamp.metadata().rangeFirstSample, 826988.6900674499, 1.0e-10);
    ASSERT_NEAR(resamp.refMetadata().chirpSlope, 588741148672.0, 1.0e-8);
    ASSERT_NEAR(resamp.doppler().coeffs[0], 301.35306906319204, 1.0e-8);

    // Allow GDAL to run Python pixel functions
    CPLSetConfigOption("GDAL_VRT_ENABLE_PYTHON", "YES");

    // Perform resampling with default lines per tile
    resamp.resamp(
        "../../data/envisat.slc.vrt",
        "warped.slc",
        "../../data/offsets/range.off",
        "../../data/offsets/azimuth.off");

    // Set lines per tile to be a weird multiple of the number of output lines
    resamp.linesPerTile(249);
    // Re-run resamp
    resamp.resamp(
        "../../data/envisat.slc.vrt",
        "warped.slc",
        "../../data/offsets/range.off",
        "../../data/offsets/azimuth.off");
}

// Compute sum of difference between reference image and warped image
TEST_F(ResampSlcTest, Validate) {
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

// Read metadata from a VRT file and return a stringstream object
std::stringstream streamFromVRT(const char * filename, int bandNum) {
    // Register GDAL drivers
    GDALAllRegister();
    // Open the VRT dataset
    GDALDataset * dataset = (GDALDataset *) GDALOpen(filename, GA_ReadOnly);
    if (dataset == NULL) {
        std::cout << "Cannot open dataset " << filename << std::endl;
        exit(1);
    }
    // Read the metadata
    char **metadata_str = dataset->GetRasterBand(bandNum)->GetMetadata("xml:isce");
    // The cereal-relevant XML is the first element in the list
    std::string meta{metadata_str[0]};
    // Close the VRT dataset
    GDALClose(dataset);
    // Convert to stream
    std::stringstream metastream;
    metastream << meta;
    // All done
    return metastream;
}

// end of file
