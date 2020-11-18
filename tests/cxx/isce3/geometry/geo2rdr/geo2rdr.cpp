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

// isce3::core
#include "isce3/core/Constants.h"
#include "isce3/core/Serialization.h"

// isce3::io
#include "isce3/io/Raster.h"

// isce3::product
#include "isce3/product/Product.h"

// isce3::geometry
#include "isce3/geometry/Geo2rdr.h"

TEST(Geo2rdrTest, RunGeo2rdr) {

    // Open the HDF5 product
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Load the product
    isce3::product::Product product(file);

    // Create geo2rdr instance
    isce3::geometry::Geo2rdr geo(product, 'A', true);

    // Load topo processing parameters to finish configuration
    geo.threshold(1e-9);
    geo.numiter(50);

    // Open topo raster from topo unit test
    isce3::io::Raster topoRaster("../topo/topo.vrt");

    // Run geo2rdr
    geo.geo2rdr(topoRaster, ".");

}

// Results should be very close to zero
TEST(Geo2rdrTest, CheckResults) {
    // Open rasters
    isce3::io::Raster rgoffRaster("range.off");
    isce3::io::Raster azoffRaster("azimuth.off");
    double rg_error = 0.0;
    double az_error = 0.0;
    for (size_t i = 0; i < rgoffRaster.length(); ++i) {
        for (size_t j = 0; j < rgoffRaster.width(); ++j) {
            // Get the offset values
            double rgoff, azoff;
            rgoffRaster.getValue(rgoff, j, i);
            azoffRaster.getValue(azoff, j, i);
            // Skip null values
            if (not std::isnan(rgoff)) {
                rg_error += rgoff*rgoff;
            }
            if (not std::isnan(azoff)) {
                az_error += azoff*azoff;
            }
        }
    }
    // Check errors; azimuth errors tend to be a little larger
    EXPECT_LT(rg_error, 1e-9);
    EXPECT_LT(az_error, 1e-9);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
