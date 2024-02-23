//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2018
//

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

// isce3::io
#include <isce3/io/IH5.h>

// isce3::core
#include <isce3/core/Constants.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Serialization.h>
#include <isce3/core/TimeDelta.h>

// isce3::product
#include <isce3/product/RadarGridProduct.h>

// isce3::geometry
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/geo2rdr_roots.h>
#include <isce3/geometry/geometry.h>

// Declaration for utility function to read test data
void loadTestData(std::vector<std::string>& aztimes,
        std::vector<double>& ranges, std::vector<double>& heights,
        std::vector<double>& ref_data, std::vector<double>& ref_zerodop);

struct GeometryTest : public ::testing::Test {

    // isce3::core objects
    isce3::core::Ellipsoid ellipsoid;
    isce3::core::LUT2d<double> doppler;
    isce3::core::Orbit orbit;

    // isce3::product objects
    isce3::product::ProcessingInformation proc;
    isce3::product::Swath swath;

    isce3::core::LookSide lookSide;

    // Constructor
protected:
    GeometryTest()
    {
        // Open the HDF5 product
        std::string h5file(TESTDATA_DIR "envisat.h5");
        isce3::io::IH5File file(h5file);

        // Instantiate a RadarGridProduct
        isce3::product::RadarGridProduct product(file);

        // Extract core and product objects
        orbit = product.metadata().orbit();
        proc = product.metadata().procInfo();
        swath = product.swath('A');
        doppler = proc.dopplerCentroid('A');
        lookSide = product.lookSide();
        ellipsoid.a(isce3::core::EarthSemiMajorAxis);
        ellipsoid.e2(isce3::core::EarthEccentricitySquared);

        // For this test, use biquintic interpolation for Doppler LUT
        doppler.interpMethod(isce3::core::BIQUINTIC_METHOD);
    }
};

TEST_F(GeometryTest, RdrToGeoWithOrbit)
{

    // Load test data
    std::vector<std::string> aztimes;
    std::vector<double> ranges, heights, ref_data, ref_zerodop;
    loadTestData(aztimes, ranges, heights, ref_data, ref_zerodop);

    // Loop over test data
    const double degrees = 180.0 / M_PI;
    for (size_t i = 0; i < aztimes.size(); ++i) {

        // Make azimuth time in seconds
        isce3::core::DateTime azDate(aztimes[i]);
        const double azTime =
                (azDate - orbit.referenceEpoch()).getTotalSeconds();

        // Evaluate Doppler
        const double dopval = doppler.eval(azTime, ranges[i]);

        // Make constant DEM interpolator set to input height
        isce3::geometry::DEMInterpolator dem(heights[i]);

        // Initialize guess
        isce3::core::cartesian_t targetLLH = {0.0, 0.0, heights[i]};

        // Run rdr2geo
        int stat = isce3::geometry::rdr2geo(azTime, ranges[i], dopval, orbit,
                ellipsoid, dem, targetLLH, swath.processedWavelength(),
                lookSide, 1.0e-8, 25, 15);

        // Check
        ASSERT_EQ(stat, 1);
        ASSERT_NEAR(degrees * targetLLH[0], ref_data[3 * i], 1.0e-8);
        ASSERT_NEAR(degrees * targetLLH[1], ref_data[3 * i + 1], 1.0e-8);
        ASSERT_NEAR(targetLLH[2], ref_data[3 * i + 2], 1.0e-8);

        // Run again with zero doppler
        stat = isce3::geometry::rdr2geo(azTime, ranges[i], 0.0, orbit,
                ellipsoid, dem, targetLLH, swath.processedWavelength(),
                lookSide, 1.0e-8, 25, 15);
        // Check
        ASSERT_EQ(stat, 1);
        ASSERT_NEAR(degrees * targetLLH[0], ref_zerodop[3 * i], 1.0e-8);
        ASSERT_NEAR(degrees * targetLLH[1], ref_zerodop[3 * i + 1], 1.0e-8);
        ASSERT_NEAR(targetLLH[2], ref_zerodop[3 * i + 2], 1.0e-8);
    }
}

TEST_F(GeometryTest, GeoToRdr)
{

    // Make a test LLH
    const double radians = M_PI / 180.0;
    isce3::core::cartesian_t llh = {
            -115.72466801139711 * radians, 34.65846532785868 * radians, 1772.0};

    // Run geo2rdr
    double aztime, slantRange;
    int stat = isce3::geometry::geo2rdr(llh, ellipsoid, orbit, doppler, aztime,
            slantRange, swath.processedWavelength(), lookSide, 1.0e-10, 50,
            10.0);
    // Convert azimuth time to a date
    isce3::core::DateTime azdate = orbit.referenceEpoch() + aztime;

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:33.993088889");
    ASSERT_NEAR(slantRange, 830450.1859446081, 1.0e-6);

    // Run geo2rdr again with zero doppler
    isce3::core::LUT2d<double> zeroDoppler;
    stat = isce3::geometry::geo2rdr(llh, ellipsoid, orbit, zeroDoppler, aztime,
            slantRange, swath.processedWavelength(), lookSide, 1.0e-10, 50,
            10.0);
    azdate = orbit.referenceEpoch() + aztime;

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:34.122893704");
    ASSERT_NEAR(slantRange, 830449.6727720434, 1.0e-6);

    // Repeat with bracketing algorithm.
    auto xyz = ellipsoid.lonLatToXyz(llh);
    stat = isce3::geometry::geo2rdr_bracket(xyz, orbit, zeroDoppler, aztime,
            slantRange, swath.processedWavelength(), lookSide, 1e-10);
    azdate = orbit.referenceEpoch() + aztime;

    EXPECT_EQ(stat, 1);
    EXPECT_EQ(azdate.isoformat(), "2003-02-26T17:55:34.122893704");
    EXPECT_NEAR(slantRange, 830449.6727720434, 1.0e-6);

    // Repeat with custom bracket.  The default run above searches the whole
    // 600 seconds of orbit data, which takes 6 iterations.  The run below
    // with a 2 second interval converges in 3 iterations.
    double t0 = aztime - 1, t1 = aztime + 1;
    stat = isce3::geometry::geo2rdr_bracket(xyz, orbit, zeroDoppler, aztime,
            slantRange, swath.processedWavelength(), lookSide, 1e-10, t0, t1);
    azdate = orbit.referenceEpoch() + aztime;

    EXPECT_EQ(stat, 1);
    EXPECT_EQ(azdate.isoformat(), "2003-02-26T17:55:34.122893704");
    EXPECT_NEAR(slantRange, 830449.6727720434, 1.0e-6);
}

TEST(Geometry, SrLkvHeadDemNed)
{
    using namespace isce3::geometry;
    // common vars and constants
    isce3::core::Ellipsoid wgs84(6378137.0, 0.0066943799901);
    const double abs_err {1e-4};
    const double d2r {M_PI / 180.0};
    const double r2d {1.0 / d2r};
    const double dem_hgt {800.0}; // (m)
    const double hgt_err {0.5};   // (m)
    const int num_iter {10};
    isce3::core::Vec3 sc_pos_ecf, sc_vel_ecf, sc_pos_llh, pnt_ecf;
    isce3::core::Vec3 est_loc_ecf, est_loc_llh, est_sc_vel_ned;
    // Note that the estimated/expected values are provided by using
    // similar modules in REE (radar echo emulator) tool
    // https://github.jpl.nasa.gov/SALSA-REE
    // clang-format off
  //(m,m,m)
  sc_pos_ecf <<
    -2434573.803881911095,
    -4820642.065286534838,
    4646722.940369521268;
  //(m/s,m/s,m/s)
  sc_vel_ecf <<
    522.995925360679,
    5107.808531616465,
    5558.156209869601;
  //(rad,rad,m)
  sc_pos_llh <<
    -116.795192003152*d2r,
    40.879509088888*d2r,
    755431.529907600489;
  //(-,-,-)
  pnt_ecf <<
    -0.292971223572,
    0.707071773397,
    -0.643597210547;
  //(m,m,m)
  est_loc_ecf <<
    -2721427.049702467397,
    -4128335.733102159109,
    4016565.662138461601;
  //(deg,deg,m)
  est_loc_llh <<
    -123.3931,
    39.2757,
    799.7505;
  //(m/s,m/s,m/s)
  est_sc_vel_ned <<
    7340.716338644244,
    -1835.775025245533,
    -12.119371107411;

    // clang-format on

    // "heading" method
    auto hdg {heading(sc_pos_llh(0), sc_pos_llh(1), sc_vel_ecf)};
    EXPECT_NEAR(hdg * r2d, -14.04062120, abs_err)
            << "Wrong heading/track angle!";

    // "slantRangeFromLookVec" method
    auto sr {slantRangeFromLookVec(sc_pos_ecf, pnt_ecf, wgs84)};
    EXPECT_NEAR(sr, 980198.86396957, abs_err)
            << "Wrong slant range at ref ellipsoid";

    // "srPosFromLookVecDem" method
    isce3::core::Vec3 loc_ecf, loc_llh;
    double sr_dem;
    auto hgt_info = srPosFromLookVecDem(sr_dem, loc_ecf, loc_llh, sc_pos_ecf,
            pnt_ecf, DEMInterpolator(dem_hgt), hgt_err, num_iter, wgs84);
    EXPECT_LE(hgt_info.first, num_iter)
            << "Wrong number of iterations for DEM height";
    EXPECT_LE(hgt_info.second, hgt_err) << "Wrong height error for DEM height";
    EXPECT_NEAR(sr_dem, 979117.2, hgt_err) << "Wrong slant range at DEM height";
    EXPECT_NEAR((loc_ecf - est_loc_ecf).cwiseAbs().maxCoeff(), 0.0, hgt_err)
            << "Wrong ECEF location at DEM height";
    EXPECT_NEAR(
            (r2d * loc_llh.head(2) - est_loc_llh.head(2)).cwiseAbs().maxCoeff(),
            0.0, abs_err)
            << "Wrong (Lon,lat) location at DEM height";
    EXPECT_NEAR(std::abs(loc_llh(2) - est_loc_llh(2)), 0.0, hgt_err)
            << "Wrong (Lon,lat) location at DEM height";

    // use mean DEM in place of optional arg in "srPosFromLookVecDem"
    auto hgt_mean_info = srPosFromLookVecDem(sr_dem, loc_ecf, loc_llh,
            sc_pos_ecf, pnt_ecf, DEMInterpolator(dem_hgt), hgt_err, num_iter,
            wgs84, dem_hgt);
    EXPECT_LE(hgt_mean_info.first, num_iter)
            << "Wrong number of iterations for mean DEM height";
    EXPECT_LE(hgt_mean_info.second, hgt_err)
            << "Wrong height error for mean DEM height";
    EXPECT_NEAR(sr_dem, 979117.2, hgt_err)
            << "Wrong slant range at mean DEM height";
    EXPECT_NEAR((loc_ecf - est_loc_ecf).cwiseAbs().maxCoeff(), 0.0, hgt_err)
            << "Wrong ECEF location at mean DEM height";
    EXPECT_NEAR(
            (r2d * loc_llh.head(2) - est_loc_llh.head(2)).cwiseAbs().maxCoeff(),
            0.0, abs_err)
            << "Wrong (Lon,lat) location at mean DEM height";
    EXPECT_NEAR(std::abs(loc_llh(2) - est_loc_llh(2)), 0.0, hgt_err)
            << "Wrong (Lon,lat) location at mean DEM height";

    // "necVector" , "nuwVector" , "enuVector"  methods
    auto sc_vel_ned {nedVector(sc_pos_llh(0), sc_pos_llh(1), sc_vel_ecf)};
    EXPECT_NEAR(
            (sc_vel_ned - est_sc_vel_ned).cwiseAbs().maxCoeff(), 0.0, abs_err)
            << "Wrong S/C Vel in NED";
    auto sc_vel_nwu {nwuVector(sc_pos_llh(0), sc_pos_llh(1), sc_vel_ecf)};
    isce3::core::Vec3 est_sc_vel_nwu {est_sc_vel_ned};
    est_sc_vel_nwu.tail(2) *= -1;
    EXPECT_NEAR(
            (sc_vel_nwu - est_sc_vel_nwu).cwiseAbs().maxCoeff(), 0.0, abs_err)
            << "Wrong S/C Vel in NWU";
    auto sc_vel_enu {enuVector(sc_pos_llh(0), sc_pos_llh(1), sc_vel_ecf)};
    isce3::core::Vec3 est_sc_vel_enu;
    est_sc_vel_enu << est_sc_vel_ned(1), est_sc_vel_ned(0), -est_sc_vel_ned(2);
    EXPECT_NEAR(
            (sc_vel_enu - est_sc_vel_enu).cwiseAbs().maxCoeff(), 0.0, abs_err)
            << "Wrong S/C Vel in ENU";
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Load test data
void loadTestData(std::vector<std::string>& aztimes,
        std::vector<double>& ranges, std::vector<double>& heights,
        std::vector<double>& ref_data, std::vector<double>& ref_zerodop)
{

    // Load azimuth times and slant ranges
    std::ifstream ifid("input_data.txt");
    std::string line;
    while (std::getline(ifid, line)) {
        std::stringstream stream;
        std::string aztime;
        double range, h;
        stream << line;
        stream >> aztime >> range >> h;
        aztimes.push_back(aztime);
        ranges.push_back(range);
        heights.push_back(h);
    }
    ifid.close();

    // Load test data for non-zero doppler
    ifid = std::ifstream("output_data.txt");
    while (std::getline(ifid, line)) {
        std::stringstream stream;
        double lat, lon, h;
        stream << line;
        stream >> lat >> lon >> h;
        ref_data.push_back(lon);
        ref_data.push_back(lat);
        ref_data.push_back(h);
    }
    ifid.close();

    // Load test data for zero doppler
    ifid = std::ifstream("output_data_zerodop.txt");
    while (std::getline(ifid, line)) {
        std::stringstream stream;
        double lat, lon, h;
        stream << line;
        stream >> lat >> lon >> h;
        ref_zerodop.push_back(lon);
        ref_zerodop.push_back(lat);
        ref_zerodop.push_back(h);
    }
    ifid.close();

    // Check sizes
    if (aztimes.size() != (ref_data.size() / 3)) {
        std::cerr << "Incompatible data sizes" << std::endl;
        exit(1);
    }
    if (aztimes.size() != (ref_zerodop.size() / 3)) {
        std::cerr << "Incompatible data sizes" << std::endl;
        exit(1);
    }
}

// end of file
