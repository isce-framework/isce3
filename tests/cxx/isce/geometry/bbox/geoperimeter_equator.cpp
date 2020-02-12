//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019
//

#include <iostream>
#include <cstdio>
#include <string>
#include <sstream>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

// isce::core
#include <isce/core/Constants.h>
#include <isce/core/DateTime.h>
#include <isce/core/Orbit.h>
#include <isce/core/LUT1d.h>
#include <isce/core/LUT2d.h>
#include <isce/core/StateVector.h>
#include <isce/core/Projections.h>

// isce::product
#include <isce/product/RadarGridParameters.h>

// isce::geometry
#include <isce/geometry/DEMInterpolator.h>
#include <isce/geometry/boundingbox.h>

using isce::core::LookSide;

struct PerimeterTest : public ::testing::TestWithParam< std::tuple<LookSide,int,int>> {

    //Reference epoch
    isce::core::DateTime t0;

    // isce::product objects
    isce::product::RadarGridParameters grid;
    // isce::core objects
    isce::core::Ellipsoid ellipsoid;
    isce::core::Orbit orbit;
    double hsat;

    // Constructor
    PerimeterTest(): t0("2017-02-12T01:12:30.0"), grid(15.0,
                                            0.06, 1000., 800000.,
                                            1000., LookSide::Right, 10000, 80,
                                            t0)
    {}

    void Setup_orbit(double lon0, double omega, int Nvec)
    {
        //WGS84 ellipsoid
        ellipsoid = isce::core::Ellipsoid(6378137.,.0066943799901);

        //Satellite height
        hsat = 700000.0;

        //Setup orbit
        orbit.referenceEpoch(t0);

        std::vector<isce::core::StateVector> statevecs(Nvec);
        for(int ii=0; ii < Nvec; ii++)
        {
            double deltat = ii * 10.0;
            double lon = lon0 + omega * deltat;
            isce::core::Vec3 pos, vel;

            pos[0] = (ellipsoid.a() + hsat) * std::cos(lon);
            pos[1] = (ellipsoid.a() + hsat) * std::sin(lon);
            pos[2] = 0.0;

            vel[0] = -omega * pos[1];
            vel[1] = omega * pos[0];
            vel[2] = 0.0;

            isce::core::DateTime epoch = t0 + deltat;

            statevecs[ii].datetime = epoch;
            statevecs[ii].position = pos;
            statevecs[ii].velocity = vel;
        }
        orbit.setStateVectors(statevecs);
    }


    //Setup the grid
    void Setup_grid(size_t azlooks, size_t rglooks, LookSide lookside)
    {
        //Customizable parameters
        grid.lookSide(lookside);
        grid.length(grid.length()/azlooks);
        grid.width(grid.width()/rglooks);
        grid.prf( grid.prf()/azlooks);
        grid.rangePixelSpacing( grid.rangePixelSpacing() * rglooks);
    }


    //Solve for Geocentric latitude given a slant range
    double solve(double R)
    {
        double temp = 1.0 + hsat/ellipsoid.a();
        double temp1 = R/ellipsoid.a();
        double A = ellipsoid.e2();
        double B = - 2 * temp;
        double C = temp*temp + 1.0 - ellipsoid.e2()- temp1*temp1;

        //Solve quadratic equation
        double D = std::sqrt(B * B - 4 * A * C);

        double x1 = (D-B)/(2*A);
        double x2 = -(D+B)/(2*A);

        double x = ( std::abs(x1) > std::abs(x2)) ? x2 : x1;
        return x;

    }
};

TEST_P(PerimeterTest, Normal) {

    LookSide side = std::get<0>(GetParam());
    int azlooks = std::get<1>(GetParam());
    int rglooks = std::get<2>(GetParam());

    // Loop over test data
    const double degrees = 180.0 / M_PI;

    //Moving at 0.1 degrees / sec
    const double lon0 = 0.0;
    const double omega = 0.1/degrees;
    const int Nvec = 10;

    //Setup orbit
    Setup_orbit(lon0, omega, Nvec);

    //Set up grid
    Setup_grid(azlooks, rglooks,side);

    //Setup projection system
    isce::core::ProjectionBase *proj = isce::core::createProj(4326);

    //Number of points per edge - default value
    int nPtsPerEdge = 11;

    //Compute perimeter
    isce::geometry::Perimeter perimeter = isce::geometry::getGeoPerimeter(grid, orbit,
                                                proj);

    //Check length of perimeter for default edge length 
    ASSERT_EQ( perimeter.getNumPoints(), 4 * nPtsPerEdge - 4 + 1);   

    //Compute slant ranges
    std::vector<double> ranges;
    std::vector<double> times;
    for (int ii=0; ii < nPtsPerEdge; ii++)
    {
        ranges.push_back( grid.slantRange( (ii/(nPtsPerEdge - 1.0)) * (grid.width()-1)));
        times.push_back( grid.sensingTime( (ii/(nPtsPerEdge - 1.0)) * (grid.length()-1)));
    }


    std::vector< std::tuple<double,double> > pts;


    //Test top edge
    for (int ii=0; ii<nPtsPerEdge; ii++)
    {
        double tinp = times[0];
        double rng = ranges[ii];
        pts.push_back( std::make_tuple(tinp, rng));
    }


    //Test far edge
    for (int ii=1; ii<nPtsPerEdge; ii++)
    {
        double tinp = times[ii];
        double rng = ranges[nPtsPerEdge-1];
        pts.push_back( std::make_tuple(tinp, rng));
    }

    //Test bottom edge
    for (int ii=nPtsPerEdge-2; ii >=0; ii--)
    {
        double tinp = times[nPtsPerEdge-1];
        double rng = ranges[ii];
        pts.push_back( std::make_tuple(tinp, rng));
    }

    //Test left edge
    for (int ii=nPtsPerEdge-2; ii >=0; ii--)
    {
        double tinp = times[ii];
        double rng = ranges[0];
        pts.push_back( std::make_tuple(tinp, rng));
    }

    int ii=0;
    for (auto rdrpt : pts)
    {
        double tinp = std::get<0>(rdrpt);
        double rng = std::get<1>(rdrpt);

        //Theoretical solutions
        double expectedLon = lon0 + omega * tinp;

        //Expected solution
        double geocentricLat = std::acos(solve(rng));

        //Convert geocentric coords to xyz
        isce::core::cartesian_t xyz = { ellipsoid.a() * std::cos(geocentricLat) * std::cos(expectedLon),
                                        ellipsoid.a() * std::cos(geocentricLat) * std::sin(expectedLon),
                                        ellipsoid.b() * std::sin(geocentricLat)};
        isce::core::cartesian_t expLLH;
        ellipsoid.xyzToLonLat(xyz, expLLH);

        OGRPoint pt;
        perimeter.getPoint(ii, &pt);

        // Check
        ASSERT_NEAR(pt.getX(), expLLH[0] * degrees, 1.0e-8);
        if (grid.lookSide() == LookSide::Left) {
            ASSERT_NEAR(pt.getY(), expLLH[1] * degrees, 1.0e-8);
        } else {
            ASSERT_NEAR(pt.getY(), -expLLH[1] * degrees, 1.0e-8);
        }
        ASSERT_NEAR(pt.getZ(), 0.0, 1.0e-8);
        ii++;

    }

    delete proj;

}


TEST_P(PerimeterTest, DateLine) {

    LookSide side = std::get<0>(GetParam());
    int azlooks = std::get<1>(GetParam());
    int rglooks = std::get<2>(GetParam());

    // Loop over test data
    const double degrees = 180.0 / M_PI;

    //Moving at 0.1 degrees / sec
    const double lon0 = -182.0/degrees;
    const double omega = 0.1/degrees;
    const int Nvec = 10;

    //Setup orbit
    Setup_orbit(lon0, omega, Nvec);

    //Set up grid
    Setup_grid(azlooks, rglooks,side);

    //Setup projection system
    isce::core::ProjectionBase *proj = isce::core::createProj(4326);


    //Compute perimeter
    isce::geometry::BoundingBox box= isce::geometry::getGeoBoundingBox(grid, orbit,
                                                proj);

    double lonStart = degrees * (lon0 + omega * grid.sensingStart());
    double lonStop = degrees * (lon0 + omega * grid.sensingStop());

    ASSERT_NEAR( box.MinX, lonStart + 360.0, 1.0e-9);
    ASSERT_NEAR( box.MaxX, lonStop + 360.0, 1.0e-9);

    delete proj;

}


INSTANTIATE_TEST_SUITE_P(PerimeterTests, PerimeterTest,
                        testing::Values(
                            std::make_tuple(LookSide::Right,1,1),
                            std::make_tuple(LookSide::Left,1,1),
                            std::make_tuple(LookSide::Right,3,5),
                            std::make_tuple(LookSide::Left,5,3)));


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


// end of file
