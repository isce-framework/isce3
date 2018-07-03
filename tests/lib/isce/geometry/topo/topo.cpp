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
#include "isce/io/IH5.h"
#include "isce/io/Raster.h"

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

    // Open the HDF5 product
    std::string h5file("../../data/envisat.h5");
    isce::io::IH5File file(h5file);

    // Deserialization
    isce::core::load(file, ellipsoid);
    isce::core::load(file, orbit, "POE");
    isce::core::load(file, doppler, "skew");
    isce::core::load(file, meta, "primary");

    // Create topo instance
    isce::geometry::Topo topo(ellipsoid, orbit, meta);

    // Load topo processing parameters to finish configuration
    std::ifstream xmlfid("../../data/topo.xml", std::ios::in);
    {
    cereal::XMLInputArchive archive(xmlfid);
    archive(cereal::make_nvp("Topo", topo));
    }

    // Open DEM raster
    isce::io::Raster demRaster("../../data/cropped.dem.grd");

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
        isce::io::Raster testRaster(test_dir + layers[k]);
        // Open the reference raster
        isce::io::Raster refRaster(ref_dir + layers[k]);
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

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
