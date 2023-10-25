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

// isce3::io
#include "isce3/io/Raster.h"
#include "isce3/io/IH5.h"

// isce3::core
#include "isce3/core/Basis.h"
#include "isce3/core/Constants.h"
#include "isce3/core/DateTime.h"
#include "isce3/core/Ellipsoid.h"
#include "isce3/core/Orbit.h"
#include "isce3/core/Pixel.h"
#include "isce3/core/LUT1d.h"
#include "isce3/core/Serialization.h"

// isce3::product
#include "isce3/product/RadarGridProduct.h"
#include "isce3/product/RadarGridParameters.h"

// isce3::geometry
#include "isce3/geometry/geometry.h"
#include "isce3/geometry/DEMInterpolator.h"

// isce3::cuda::geometry
#include "isce3/cuda/geometry/gpuGeometry.h"

using isce3::core::Vec3;
using isce3::geometry::DEMInterpolator;

// Declaration for function to load DEM
void loadDEM(DEMInterpolator & demInterp);

struct GpuGeometryTest : public ::testing::Test {

    // isce3::core objects
    isce3::core::Ellipsoid ellipsoid;
    isce3::core::LUT2d<double> doppler;
    isce3::core::Orbit orbit;

    // isce3::product objects
    isce3::product::RadarGridParameters rgparam;
    isce3::core::LookSide lookSide;

    // Constructor
    protected:
        GpuGeometryTest() {
            // Open the HDF5 product
            std::string h5file(TESTDATA_DIR "envisat.h5");
            isce3::io::IH5File file(h5file);

            // Instantiate a RadarGridProduct
            isce3::product::RadarGridProduct product(file);

            // Extract core and product objects
            orbit = product.metadata().orbit();
            rgparam = isce3::product::RadarGridParameters(product, 'A');
            doppler = product.metadata().procInfo().dopplerCentroid('A');
            lookSide = product.lookSide();
            ellipsoid.a(isce3::core::EarthSemiMajorAxis);
            ellipsoid.e2(isce3::core::EarthEccentricitySquared);

            // For this test, use biquintic interpolation for Doppler LUT
            doppler.interpMethod(isce3::core::BIQUINTIC_METHOD);
        }
};

TEST_F(GpuGeometryTest, RdrToGeoWithInterpolation) {

    // Load DEM subset covering test points
    DEMInterpolator dem(-500.0, isce3::core::BILINEAR_METHOD);
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
            isce3::core::cartesian_t targetLLH = {0.0, 0.0, 1000.0};

            // Interpolate orbit to get state vector
            Vec3 pos, vel;
            orbit.interpolate(&pos, &vel, azTime);

            // Setup geocentric TCN basis
            isce3::core::Basis TCNbasis(pos, vel);

            // Compute satellite velocity magnitude
            const double vmag = vel.norm();
            // Compute Doppler factor
            const double dopfact = 0.5 * rgparam.wavelength() * dopval * range / vmag;

            // Wrap range and Doppler factor in a Pixel object
            isce3::core::Pixel pixel(range, dopfact, 0);

            // Run rdr2geo on CPU
            int stat_cpu = isce3::geometry::rdr2geo(pixel, TCNbasis, pos, vel,
                ellipsoid, dem, targetLLH, lookSide, 1.0e-4, maxiter, extraiter);
            // Cache results
            const double reflon = degrees * targetLLH[0];
            const double reflat = degrees * targetLLH[1];
            const double refhgt = targetLLH[2];

            // Reset result
            targetLLH = {0.0, 0.0, 1000.0};

            // Run rdr2geo on GPU
            int stat_gpu = isce3::cuda::geometry::rdr2geo_h(pixel, TCNbasis, pos, vel,
                ellipsoid, dem, targetLLH, lookSide, 1.0e-4, maxiter, extraiter);

            // Check
            ASSERT_EQ(stat_cpu, stat_gpu);
            ASSERT_NEAR(degrees*targetLLH[0], reflon, 1.0e-8);
            ASSERT_NEAR(degrees*targetLLH[1], reflat, 1.0e-8);
            ASSERT_NEAR(targetLLH[2], refhgt, 1.0e-2);

            Vec3 targetXYZ;
            stat_gpu = isce3::cuda::geometry::rdr2geo_bracket_h(azTime, range,
                    dopval, orbit, ellipsoid, dem, targetXYZ,
                    rgparam.wavelength(), lookSide, 1e-4, 0.0, M_PI / 2);

            targetLLH = ellipsoid.xyzToLonLat(targetXYZ);
            // don't trigger failure if GPU bracket algorithm succeeds on pixels
            // where old CPU algorithm fails.
            if (stat_cpu != 0) {
                ASSERT_EQ(stat_cpu, stat_gpu);
                ASSERT_NEAR(degrees * targetLLH[0], reflon, 1.0e-8);
                ASSERT_NEAR(degrees * targetLLH[1], reflat, 1.0e-8);
                ASSERT_NEAR(targetLLH[2], refhgt, 1.0e-2);
            }
        }
    }
}

TEST_F(GpuGeometryTest, GeoToRdr) {

    // Make a reference epoch for numerical precision
    isce3::core::DateTime refEpoch(2003, 2, 25);
    orbit.referenceEpoch(refEpoch);

    // Make a test LLH
    const double radians = M_PI / 180.0;
    isce3::core::cartesian_t llh = {
        -115.72466801139711 * radians,
        34.65846532785868 * radians,
        1772.0
    };

    // Run geo2rdr on gpu
    double aztime, slantRange;
    auto doppler_1d = isce3::core::avgLUT2dToLUT1d<double>(doppler);
    int stat = isce3::cuda::geometry::geo2rdr_h(llh, ellipsoid, orbit, doppler_1d,
        aztime, slantRange, rgparam.wavelength(), rgparam.lookSide(), 1.0e-10, 50, 10.0);
    // Convert azimuth time to a date
    isce3::core::DateTime azdate = refEpoch + aztime;

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:33.993088889");
    ASSERT_NEAR(slantRange, 830450.1859446081, 1.0e-6);

    // Run geo2rdr again with zero doppler
    isce3::core::LUT1d<double> zeroDoppler;
    stat = isce3::cuda::geometry::geo2rdr_h(llh, ellipsoid, orbit, zeroDoppler,
        aztime, slantRange, rgparam.wavelength(), rgparam.lookSide(), 1.0e-10, 50, 10.0);
    azdate = refEpoch + aztime;

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:34.122893704");
    ASSERT_NEAR(slantRange, 830449.6727720434, 1.0e-6);

    // Run geo2rdr_bracket with zero doppler
    const auto xyz = ellipsoid.lonLatToXyz(llh);
    const auto zerodop2d = isce3::core::LUT2d<double>();
    stat = isce3::cuda::geometry::geo2rdr_bracket_h(xyz, orbit, zerodop2d,
        aztime, slantRange, rgparam.wavelength(), rgparam.lookSide(), 1e-8);
    azdate = refEpoch + aztime;

    EXPECT_EQ(stat, 1);
    EXPECT_EQ(azdate.isoformat(), "2003-02-26T17:55:34.122893704");
    EXPECT_NEAR(slantRange, 830449.6727720434, 1.0e-6);

    // Run geo2rdr_bracket again using custom bracket
    double t0 = aztime - 1, t1 = aztime + 1;
    stat = isce3::cuda::geometry::geo2rdr_bracket_h(xyz, orbit, zerodop2d,
        aztime, slantRange, rgparam.wavelength(), rgparam.lookSide(), 1e-8,
        t0, t1);
    azdate = refEpoch + aztime;

    EXPECT_EQ(stat, 1);
    EXPECT_EQ(azdate.isoformat(), "2003-02-26T17:55:34.122893704");
    EXPECT_NEAR(slantRange, 830449.6727720434, 1.0e-6);
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

    // Open DEM raster
    isce3::io::Raster demRaster(TESTDATA_DIR "srtm_cropped.tif");

    // Extract DEM subset
    demInterp.loadDEM(demRaster, min_lon, max_lon, min_lat, max_lat);
}

// end of file
