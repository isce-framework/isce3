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
    isce::core::load(file, doppler, "skew_dcpolynomial");
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
    isce::io::Raster demRaster("../../data/srtm_cropped.tif");

    // Run topo
    topo.topo(demRaster, doppler, ".");

}

TEST(TopoTest, CheckResults) {
    
    // Open generated topo raster
    isce::io::Raster testRaster("topo.vrt");
    
    // Open reference topo raster
    isce::io::Raster refRaster("../../data/topo/topo.vrt");

    // The associated tolerances
    std::vector<double> tols{1.0e-5, 1.0e-5, 0.15, 1.0e-4, 1.0e-4, 0.02, 0.02};

    // The directories where the data are
    std::string test_dir = "./";
    std::string ref_dir = "../../data/topo/";

    // Valarrays to hold line of data
    std::valarray<double> test(testRaster.width()), ref(refRaster.width());

    // Loop over topo bands
    for (size_t k = 0; k < refRaster.numBands(); ++k) {
        // Compute sum of absolute error
        double error = 0.0;
        size_t count = 0;
        for (size_t i = 0; i < testRaster.length(); ++i) {
            // Get line of data
            testRaster.getLine(test, i, k + 1);
            refRaster.getLine(ref, i, k + 1);
            for (size_t j = 0; j < testRaster.width(); ++j) {
                // Get the values
                const double testVal = test[j];
                const double refVal = ref[j];
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
