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
#include "isce3/product/RadarGridProduct.h"

// isce3::geometry
#include "isce3/geometry/Geo2rdr.h"

TEST(Geo2rdrTest, RunGeo2rdr) {

    // Open the HDF5 product
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Load the product
    isce3::product::RadarGridProduct product(file);

    // Create geo2rdr instance
    isce3::geometry::Geo2rdr geo(product, 'A', true);

    // Load topo processing parameters to finish configuration
    auto threshold = 1e-6 / geo.radarGridParameters().prf();
    geo.threshold(threshold);
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
            // Accumulate error
            if (std::abs(rgoff) > 999.0 || std::abs(azoff) > 999.0)
                continue;
            // Accumulate error
            rg_error = std::max(rg_error, std::abs(rgoff));
            az_error = std::max(az_error, std::abs(azoff));
        }
    }
    // Check errors.  The Newton step size isn't a perfect estimate of the error
    // of the solution, so allow geo2rdr a little wiggle room in excess of the
    // requested 1e-6 pixel error.
    EXPECT_LT(az_error, 2e-6);
    // In general there's no reason to expect the same convergence tolerance
    // in range.  But rg_error approaches zero as the Doppler centroid
    // approaches zero, so this check is okay given our particular test.
    EXPECT_LT(rg_error, 2e-6);
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
