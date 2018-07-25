// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Liang Yu
// Copyright 2018
//

#include <cmath>
#include <iostream>
#include <vector>
#include "isce/core/Constants.h"
#include "isce/core/Orbit.h"
#include "isce/core/cuda/gpuOrbit.h"
#include "gtest/gtest.h"

using isce::core::orbitInterpMethod;
using isce::core::HERMITE_METHOD;
using isce::core::LEGENDRE_METHOD;
using isce::core::SCH_METHOD;
using isce::core::Orbit;
using isce::core::cuda::gpuOrbit;
using std::endl;
using std::vector;
using isce::core::cartesian_t;

struct OrbitTest : public ::testing::Test {
    virtual void SetUp() {
        fails = 0;
    }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "gpuOrbit::TearDown sees failures" << std::endl;
        }
    }
    unsigned fails;
};

#define compareTriplet(a,b,c)\
    EXPECT_NEAR(a[0], b[0], c); \
    EXPECT_NEAR(a[1], b[1], c); \
    EXPECT_NEAR(a[2], b[2], c);

void makeLinearSV(double dt, cartesian_t &opos, cartesian_t &ovel, cartesian_t &pos,
                  cartesian_t &vel) {
    pos = {opos[0] + (dt * ovel[0]), opos[1] + (dt * ovel[1]), opos[2] + (dt * ovel[2])};
    vel = ovel;
}

TEST_F(OrbitTest, LinearSCH) {
    /*
     * Test linear orbit.
     */

    Orbit orb(1,11);
    double t = 1000.;
    cartesian_t opos = {0., 0., 0.};
    cartesian_t ovel = {4000., -1000., 4500.};
    cartesian_t pos, vel;
    cartesian_t hpos, hvel;

    // Create straight-line orbit with 11 state vectors, each 10 s apart
    for (int i=0; i<11; i++) {
        makeLinearSV(i*10., opos, ovel, pos, vel); 
        orb.setStateVector(i, t+(i*10.), pos, vel);
    }

    // Interpolation test times
    double test_t[] = {23.3, 36.7, 54.5, 89.3};
    cartesian_t ref_pos, ref_vel;

    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    for (int i=0; i<4; i++) {
        makeLinearSV(test_t[i], opos, ovel, ref_pos, ref_vel);
        orb.interpolate(t+test_t[i], pos, vel, SCH_METHOD);
        orb.interpolate(t+test_t[i], hpos, hvel, HERMITE_METHOD);
        compareTriplet(pos, ref_pos, 1.0e-5);
        compareTriplet(vel, ref_vel, 1.0e-6);
    }

    fails += ::testing::Test::HasFailure();
}

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
