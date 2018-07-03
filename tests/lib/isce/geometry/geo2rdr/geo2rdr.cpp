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
#include "isce/core/Serialization.h"

// isce::io
#include "isce/io/Raster.h"

// isce::geometry
#include "isce/geometry/Serialization.h"
#include "isce/geometry/Geo2rdr.h"

// Declaration for utility function to read metadata stream from VRT
std::stringstream streamFromVRT(const char * filename, int bandNum=1);

TEST(Geo2rdrTest, RunGeo2rdr) {

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

    // Create geo2rdr isntance
    isce::geometry::Geo2rdr geo(ellipsoid, orbit, meta);

    // Load topo processing parameters to finish configuration
    std::ifstream xmlfid("../../data/topo.xml", std::ios::in);
    {
    cereal::XMLInputArchive archive(xmlfid);
    archive(cereal::make_nvp("Geo2rdr", geo));
    }

    // Open topo raster from topo unit test
    isce::io::Raster topoRaster("../topo/topo.vrt");

    // Run geo2rdr
    geo.geo2rdr(topoRaster, doppler, ".");

}

// Results should be very close to zero
TEST(Geo2rdrTest, CheckResults) {
    // Open rasters
    isce::io::Raster rgoffRaster("range.off");
    isce::io::Raster azoffRaster("azimuth.off");
    double rg_error = 0.0;
    double az_error = 0.0;
    for (size_t i = 0; i < rgoffRaster.length(); ++i) {
        for (size_t j = 0; j < rgoffRaster.width(); ++j) {
            // Get the offset values
            double rgoff, azoff;
            rgoffRaster.getValue(rgoff, j, i);
            azoffRaster.getValue(azoff, j, i);
            // Skip null values
            if (std::abs(rgoff) > 999.0 || std::abs(azoff) > 999.0)
                continue;
            // Accumulate error
            rg_error += rgoff*rgoff;
            az_error += azoff*azoff;
        }
    }
    // Check errors; azimuth errors tend to be a little larger
    ASSERT_TRUE(rg_error < 1.0e-10);
    ASSERT_TRUE(az_error < 1.0e-10);
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
