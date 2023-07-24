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
#include "isce3/io/IH5.h"
#include "isce3/io/Raster.h"

// isce3::product
#include "isce3/product/RadarGridProduct.h"

// isce3::cuda::geometry
#include "isce3/cuda/geometry/Topo.h"

// Declaration for utility function to read metadata stream from VRT
std::stringstream streamFromVRT(const char * filename, int bandNum=1);

TEST(GPUTopoTest, RunTopo) {

    // Open the HDF5 product
    std::string h5file(TESTDATA_DIR "envisat.h5");
    isce3::io::IH5File file(h5file);

    // Load the product
    isce3::product::RadarGridProduct product(file);

    // Create topo instance
    isce3::cuda::geometry::Topo topo(product, 'A', true);

    // Load topo processing parameters to finish configuration
    topo.threshold(0.05);
    topo.numiter(25);
    topo.extraiter(10);
    topo.demMethod(isce3::core::dataInterpMethod::BIQUINTIC_METHOD);
    topo.epsgOut(4326);

    // Open DEM raster
    isce3::io::Raster demRaster(TESTDATA_DIR "srtm_cropped.tif");

    // Run topo
    topo.topo(demRaster, ".");

}

TEST(GPUTopoTest, CheckResults) {

    // Open generated topo raster
    isce3::io::Raster testRaster("topo.vrt");

    // Open reference topo raster
    isce3::io::Raster refRaster(TESTDATA_DIR "topo/topo.vrt");

    // The associated tolerances
    std::vector<double> tols{1.0e-5, 1.0e-5, 0.15, 1.0e-4, 1.0e-4, 0.02, 0.02};

    // The directories where the data are
    std::string test_dir = "./";
    std::string ref_dir = TESTDATA_DIR "topo/";

    // Valarrays to hold line of data
    std::valarray<double> test(testRaster.width()), ref(refRaster.width());

    // Loop over reference topo bands - discounting groundToSatEast,
    // groundToSatNorth, azimuth as there is no reference test data to compare
    // against.
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
