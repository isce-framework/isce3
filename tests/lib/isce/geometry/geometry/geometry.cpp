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
#include <gtest/gtest.h>

// isce::core
#include "isce/core/Constants.h"
#include "isce/core/DateTime.h"
#include "isce/core/Ellipsoid.h"
#include "isce/core/LinAlg.h"
#include "isce/core/Orbit.h"
#include "isce/core/Peg.h"
#include "isce/core/Pegtrans.h"
#include "isce/core/Poly2d.h"
#include "isce/core/Metadata.h"
#include "isce/core/Serialization.h"

// isce::geometry
#include "isce/geometry/Geometry.h"

// Declaration for utility function to read metadata stream from VRT
std::stringstream streamFromVRT(const char * filename, int bandNum=1);

struct GeometryTest : public ::testing::Test {

    // isce::core objects
    isce::core::DateTime azDate;
    isce::core::Ellipsoid ellipsoid;
    isce::core::Peg peg;
    isce::core::Pegtrans ptm;
    isce::core::Poly2d contentDoppler;
    isce::core::Poly2d skewDoppler;
    isce::core::Metadata meta;
    isce::core::Orbit orbit;
    isce::core::StateVector state;

    // isce::geometry objects
    isce::geometry::Pixel pixel;
    isce::geometry::Basis basis;

    // Constructor
    protected:
        GeometryTest() {
            // Load metadata stream
            std::stringstream metastream = streamFromVRT("../../data/envisat.slc.vrt");

            // Deserialization
            {
            cereal::XMLInputArchive archive(metastream);
            archive(cereal::make_nvp("ContentDoppler", contentDoppler),
                    cereal::make_nvp("SkewDoppler", skewDoppler),
                    cereal::make_nvp("Ellipsoid", ellipsoid),
                    cereal::make_nvp("Orbit", orbit),
                    cereal::make_nvp("Radar", meta));
            }

            // Interpolate orbit
            const isce::core::DateTime azDate("2003-02-26T17:55:28.00");
            const double azTime = azDate.secondsSinceEpoch();
            isce::core::cartesian_t xyzsat, velsat;
            int stat = orbit.interpolate(azTime, xyzsat, velsat, isce::core::HERMITE_METHOD);
            if (stat != 0) {
                std::cerr << "Unable to interpolate orbit." << std::endl;
            }
            state.position(xyzsat);
            state.velocity(velsat);
            const double satVmag = isce::core::LinAlg::norm(velsat);

            // Get TCN basis
            isce::core::cartesian_t that, chat, nhat;
            ellipsoid.TCNbasis(xyzsat, velsat, that, chat, nhat);
            basis.that(that);   
            basis.chat(chat);
            basis.nhat(nhat);

            // Set peg point right below satellite
            isce::core::cartesian_t llhsat;
            ellipsoid.xyzToLatLon(xyzsat, llhsat);
            peg.lat = llhsat[0];
            peg.lon = llhsat[1];
            //peg.hdg = meta.pegHeading;
            peg.hdg = -166.40653160564963 * M_PI / 180.0;
            ptm.radarToXYZ(ellipsoid, peg);

            // Set pixel properties
            const size_t rbin = 500;
            const double slantRange = meta.rangeFirstSample 
                                    + rbin * meta.slantRangePixelSpacing;
            pixel.range(slantRange);
            pixel.dopfact((0.5 * meta.radarWavelength * (contentDoppler.eval(0, rbin) / satVmag))
                * slantRange);
            pixel.bin(rbin);
        }
};

TEST_F(GeometryTest, RdrToGeo) {
    // Make a constant DEM interpolator
    isce::geometry::DEMInterpolator demInterp(0.0);

    // Initialize guess 
    isce::core::cartesian_t targetLLH = {0.0, 0.0, 0.0};

    // Run rdr2geo
    int stat = isce::geometry::Geometry::rdr2geo(pixel, basis, state, ellipsoid, ptm, demInterp,
        targetLLH, meta.lookSide, 1.0e-6, 25, 10);

    // Check results
    const double degrees = 180.0 / M_PI;
    ASSERT_EQ(stat, 1);
    ASSERT_NEAR(degrees * targetLLH[0], 35.005082934361745, 1.0e-8);
    ASSERT_NEAR(degrees * targetLLH[1], -115.5921003238083, 1.0e-8);
    ASSERT_NEAR(targetLLH[2], 2.835586858903558e-09, 1.0e-8);

    // Run it again with zero doppler
    pixel.dopfact(0.0);
    stat = isce::geometry::Geometry::rdr2geo(pixel, basis, state, ellipsoid, ptm, demInterp,
        targetLLH, meta.lookSide, 1.0e-6, 25, 10);
    ASSERT_EQ(stat, 1);
    ASSERT_NEAR(degrees * targetLLH[0], 35.01267683520824, 1.0e-8);
    ASSERT_NEAR(degrees * targetLLH[1], -115.59009580548408, 1.0e-8);
    ASSERT_NEAR(targetLLH[2], 0.0, 1.0e-8);
}

TEST_F(GeometryTest, GeoToRdr) {
    
    // Make a test LLH
    const double radians = M_PI / 180.0;
    isce::core::cartesian_t llh = {35.10*radians, -115.6*radians, 55.0};

    // Run geo2rdr
    double aztime, slantRange;
    int stat = isce::geometry::Geometry::geo2rdr(llh, ellipsoid, orbit, skewDoppler,
        meta, aztime, slantRange, 1.0e-8, 50, 1.0e-8);
    // Convert azimuth time to a date
    isce::core::DateTime azdate;
    azdate.secondsSinceEpoch(aztime);

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:26.487008");
    ASSERT_NEAR(slantRange, 831834.3551143121, 1.0e-6);

    // Run geo2rdr again with zero doppler
    isce::core::Poly2d zeroDoppler;
    stat = isce::geometry::Geometry::geo2rdr(llh, ellipsoid, orbit, zeroDoppler,
        meta, aztime, slantRange, 1.0e-8, 50, 1.0e-8);
    azdate.secondsSinceEpoch(aztime);

    ASSERT_EQ(stat, 1);
    ASSERT_EQ(azdate.isoformat(), "2003-02-26T17:55:26.613450");
    ASSERT_NEAR(slantRange, 831833.869159697, 1.0e-6);
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
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

// end of file
