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
#include <fstream>
#include <gtest/gtest.h>

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/Raster.h"
#include "isce/core/Serialization.h"

// isce::geometry
#include "isce/geometry/Serialization.h"
#include "isce/geometry/Topo.h"

// Declaration for utility function to read metadata stream from VRT
std::stringstream streamFromVRT(const char * filename, int bandNum=1);

TEST(TopoTest, RunTopo) {

    // Instantiate isce::core objects
    isce::core::Poly2d doppler;
    isce::core::Orbit orbit;
    isce::core::Ellipsoid ellipsoid;
    isce::core::Metadata meta;

    // Load metadata
    std::stringstream metastream = streamFromVRT("../../data/envisat.slc.vrt");
    {
    cereal::XMLInputArchive archive(metastream);
    archive(cereal::make_nvp("Orbit", orbit),
            cereal::make_nvp("SkewDoppler", doppler),
            cereal::make_nvp("Ellipsoid", ellipsoid),
            cereal::make_nvp("Radar", meta));
    }

    // Create topo instance
    isce::geometry::Topo topo(ellipsoid, orbit, meta);

    // Load topo processing parameters to finish configuration
    std::ifstream xmlfid("../../data/topo.xml", std::ios::in);
    {
    cereal::XMLInputArchive archive(xmlfid);
    archive(cereal::make_nvp("Topo", topo));
    }

    // Open DEM raster
    isce::core::Raster demRaster("../../data/srtm_cropped.tif");

    // Run topo
    topo.topo(demRaster, doppler, ".");

}

TEST(TopoTest, CheckResults) {

    // The list of files to check
    std::vector<std::string> layers{"lat.rdr", "lon.rdr", "z.rdr", "inc.rdr",
        "hdg.rdr", "localInc.rdr", "localPsi.rdr"};

    // The associated tolerances
    std::vector<double> tols{1.0e-5, 1.0e-5, 0.15, 1.0e-4, 1.0e-4, 0.02, 0.02};

    // The directories where the data are
    std::string test_dir = "./";
    std::string ref_dir = "../../data/topo/";

    // Loop over files
    for (size_t k = 0; k < layers.size(); ++k) {
        // Open the test raster
        isce::core::Raster testRaster(test_dir + layers[k]);
        // Open the reference raster
        isce::core::Raster refRaster(ref_dir + layers[k]);
        // Compute sum of absolute error
        const size_t N = testRaster.length() * testRaster.width();
        double error = 0.0;
        size_t count = 0;
        for (size_t i = 0; i < testRaster.length(); ++i) {
            for (size_t j = 0; j < testRaster.width(); ++j) {
                // Get the values
                double testVal, refVal;
                testRaster.getValue(testVal, j, i);
                refRaster.getValue(refVal, j, i);
                // Accumulate the error (skip outliers)
                const double currentError = std::abs(testVal - refVal);
                if (currentError > 5.0) continue;
                error += std::abs(testVal - refVal);
                ++count;
            }
        }
        // Normalize the error and check
        ASSERT_TRUE((error / count) < tols[k]);
    }
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

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
