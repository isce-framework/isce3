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

// isce::product
#include "isce/product/Product.h"

// isce::geometry
#include "isce/geometry/Serialization.h"
#include "isce/geometry/Geo2rdr.h"

TEST(Geo2rdrTest, RunGeo2rdr) {

    // Open the HDF5 product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Load the product
    isce::product::Product product(file);

    // Create geo2rdr instance
    isce::geometry::Geo2rdr geo(product, 'A', true);

    // Load topo processing parameters to finish configuration
    std::ifstream xmlfid("../../data/topo.xml", std::ios::in);
    {
    cereal::XMLInputArchive archive(xmlfid);
    archive(cereal::make_nvp("Geo2rdr", geo));
    }

    // Open topo raster from topo unit test
    isce::io::Raster topoRaster("../topo/topo.vrt");

    // Run geo2rdr
    geo.geo2rdr(topoRaster, ".");

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

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
