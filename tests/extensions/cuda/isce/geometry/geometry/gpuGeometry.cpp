//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2018
//

#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <gtest/gtest.h>

// isce::io
#include "isce/io/Raster.h"
#include "isce/io/IH5.h"

// isce::core
#include "isce/core/Basis.h"
#include "isce/core/Constants.h"
#include "isce/core/DateTime.h"
#include "isce/core/Ellipsoid.h"
#include "isce/core/Orbit.h"
#include "isce/core/Pixel.h"
#include "isce/core/LUT1d.h"
#include "isce/core/Serialization.h"
#include "isce/core/StateVector.h"

// isce::product
#include "isce/product/Product.h"
#include "isce/product/RadarGridParameters.h"

// isce::geometry
#include "isce/geometry/geometry.h"
#include "isce/geometry/DEMInterpolator.h"

// isce::cuda::geometry
#include "isce/cuda/geometry/gpuGeometry.h"

using isce::core::LinAlg;
using isce::geometry::DEMInterpolator;

// Declaration for function to load DEM
void loadDEM(DEMInterpolator & demInterp);

struct GpuGeometryTest : public ::testing::Test {

    // isce::core objects
    isce::core::Ellipsoid ellipsoid;
    isce::core::LUT2d<double> doppler;
    isce::core::Orbit orbit;

    // isce::product objects
    isce::product::RadarGridParameters rgparam;
    int lookSide;

    // Constructor
    protected:
        GpuGeometryTest() {
            // Open the HDF5 product
            std::string h5file("../../../../../lib/isce/data/envisat.h5");
            isce::io::IH5File file(h5file);

            // Instantiate a Product
            isce::product::Product product(file);

            // Extract core and product objects
            orbit = product.metadata().orbit();
            rgparam = isce::product::RadarGridParameters(product, 'A');
            doppler = product.metadata().procInfo().dopplerCentroid('A');
            lookSide = product.lookSide();
            ellipsoid.a(isce::core::EarthSemiMajorAxis);
            ellipsoid.e2(isce::core::EarthEccentricitySquared);

            // For this test, use biquintic interpolation for Doppler LUT
            doppler.interpMethod(isce::core::BIQUINTIC_METHOD);
        }
};

TEST_F(GpuGeometryTest, RdrToGeoWithInterpolation) {

    // Load DEM subset covering test points
    DEMInterpolator dem(-500.0, isce::core::BILINEAR_METHOD);
    loadDEM(dem);

    // Loop over uniform grid of test points
    const double degrees = 180.0 / M_PI;
    const int maxiter = 25;
    const int extraiter = 10;
    for (size_t i = 10; i < 500; i += 40) {
        for (size_t j = 10; j < 500; j += 40) {

            // Get azimutha and range info
            const double azTime = rgparam.sensingTime(i);
            const double range = rgparam.slantRange(j);
            const double dopval = doppler.eval(azTime, range);

            // Initialize guess
            isce::core::cartesian_t targetLLH = {0.0, 0.0, 1000.0};

            // Interpolate orbit to get state vector
            isce::core::StateVector state;
            int stat = orbit.interpolate(azTime, state, isce::core::HERMITE_METHOD);

            // Setup geocentric TCN basis
            isce::core::Basis TCNbasis;
            isce::geometry::geocentricTCN(state, TCNbasis);

            // Compute satellite velocity magnitude
            const double vmag = LinAlg::norm(state.velocity());
            // Compute Doppler factor
            const double dopfact = 0.5 * rgparam.wavelength() * dopval * range / vmag;

            // Wrap range and Doppler factor in a Pixel object
            isce::core::Pixel pixel(range, dopfact, 0);
                    
            // Run rdr2geo on CPU
            int stat_cpu = isce::geometry::rdr2geo(pixel, TCNbasis, state,
                ellipsoid, dem, targetLLH, lookSide, 1.0e-4, maxiter, extraiter);
            // Cache results
            const double reflon = degrees * targetLLH[0];
            const double reflat = degrees * targetLLH[1];
            const double refhgt = targetLLH[2];

            // Reset result
            targetLLH = {0.0, 0.0, 1000.0};

            // Run rdr2geo on GPU
            int stat_gpu = isce::cuda::geometry::rdr2geo_h(pixel, TCNbasis, state,
                ellipsoid, dem, targetLLH, lookSide, 1.0e-4, maxiter, extraiter);

            // Check
            ASSERT_NEAR(degrees*targetLLH[0], reflon, 1.0e-8);
            ASSERT_NEAR(degrees*targetLLH[1], reflat, 1.0e-8);
            ASSERT_NEAR(targetLLH[2], refhgt, 1.0e-2);
        }
    }
}

TEST_F(GpuGeometryTest, GeoToRdr) {

    // Make a reference epoch for numerical precision
    isce::core::DateTime refEpoch(2003, 2, 25);
    orbit.updateUTCTimes(refEpoch);

    // Make a test LLH
    const double radians = M_PI / 180.0;
    isce::core::cartesian_t llh = {
        -115.72466801139711 * radians,
        34.65846532785868 * radians,
        1772.0
    };

    // Run geo2rdr on gpu
    double aztime, slantRange;
    int stat = isce::cuda::geometry::geo2rdr_h(llh, ellipsoid, orbit, doppler,
        aztime, slantRange, rgparam.wavelength(), 1.0e-10, 50, 10.0);
    // Convert azimuth time to a date
    isce::core::DateTime azdate = refEpoch + aztime;

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:33.993088889");
    ASSERT_NEAR(slantRange, 830450.1859446081, 1.0e-6);

    // Run geo2rdr again with zero doppler
    isce::core::LUT1d<double> zeroDoppler;
    stat = isce::cuda::geometry::geo2rdr_h(llh, ellipsoid, orbit, zeroDoppler,
        aztime, slantRange, rgparam.wavelength(), 1.0e-10, 50, 10.0);
    azdate = refEpoch + aztime;

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:34.122893704");
    ASSERT_NEAR(slantRange, 830449.6727720434, 1.0e-6);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

void loadDEM(DEMInterpolator & demInterp) {

    // Bounds for DEM
    double min_lon = -115.8;
    double min_lat = 34.62;
    double max_lon = -115.32;
    double max_lat = 35.0;

    // Convert to radians
    min_lon *= M_PI / 180.0;
    min_lat *= M_PI / 180.0;
    max_lon *= M_PI / 180.0;
    max_lat *= M_PI / 180.0;

    // Open DEM raster
    isce::io::Raster demRaster("../../../../../lib/isce/data/srtm_cropped.tif"); 

    // Extract DEM subset
    demInterp.loadDEM(demRaster, min_lon, max_lon, min_lat, max_lat, demRaster.getEPSG());
}

// end of file
