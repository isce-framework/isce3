//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <iostream>
#include <vector>
#include <isce/core/Orbit.h>
#include "gtest/gtest.h"
using isce::core::orbitInterpMethod;
using isce::core::HERMITE_METHOD;
using isce::core::LEGENDRE_METHOD;
using isce::core::SCH_METHOD;
using isce::core::Orbit;
using isce::core::cartesian_t;
using std::cout;
using std::endl;
using std::vector;

struct OrbitTest : public ::testing::Test {
    virtual void SetUp() {
        fails = 0;
    }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "Orbit::TearDown sees failures" << std::endl;
        }
    }
    unsigned fails;
};


#define compareTriplet(a,b)\
    EXPECT_EQ(a[0], b[0]); \
    EXPECT_EQ(a[1], b[1]); \
    EXPECT_EQ(a[2], b[2]);


void makeLinearSV(double dt, cartesian_t &opos, cartesian_t &ovel, cartesian_t &pos,
                  cartesian_t &vel) {
    pos = {opos[0] + (dt * ovel[0]), opos[1] + (dt * ovel[1]), opos[2] + (dt * ovel[2])};
    vel = ovel;
}

TEST_F(OrbitTest,Reverse) {
    /*
     * Test linear orbit.
     */

    Orbit orb(11);
    double t = 1000.;
    double t1;
    cartesian_t opos = {0., 0., 0.};
    cartesian_t ovel = {4000., -1000., 4500.};
    cartesian_t pos, vel;

    // Create straight-line orbit with 11 state vectors, each 10 s apart
    for (int i=0; i<11; i++) {
        makeLinearSV(i*10., opos, ovel, pos, vel);
        orb.setStateVector(i, t+(i*10.), pos, vel);
    }


    Orbit newOrb(0);

    for(int i=10; i>=0; i--)
    {
        orb.getStateVector(i, t, pos, vel);
        newOrb.addStateVector(t,pos,vel);
    }

    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    for (int i=0; i<10; i++) {
        orb.getStateVector(i, t, pos, vel);
        newOrb.getStateVector(i, t1, opos, ovel);

        EXPECT_EQ(t1,t);
        compareTriplet(opos, pos);
        compareTriplet(ovel, vel);
    }

    fails += ::testing::Test::HasFailure();

}

TEST_F(OrbitTest,OutOfOrder) {
    /*
     * Test linear orbit.
     */

    Orbit orb(11);
    double t = 1000.;
    double t1;
    cartesian_t opos = {0., 0., 0.};
    cartesian_t ovel = {4000., -1000., 4500.};
    cartesian_t pos, vel;

    // Create straight-line orbit with 11 state vectors, each 10 s apart
    for (int i=0; i<11; i++) {
        makeLinearSV(i*10., opos, ovel, pos, vel);
        orb.setStateVector(i, t+(i*10.), pos, vel);
    }


    Orbit newOrb(0);

    for(int i=10; i>=0; i-=2)
    {
        orb.getStateVector(i, t, pos, vel);
        newOrb.addStateVector(t,pos,vel);
    }


    for(int i=1; i<10; i+=2)
    {
        orb.getStateVector(i, t, pos, vel);
        newOrb.addStateVector(t, pos, vel);
    }


    // Test each interpolation time against SCH, Hermite, and Legendre interpolation methods
    for (int i=0; i<10; i++) {
        orb.getStateVector(i, t, pos, vel);
        newOrb.getStateVector(i, t1, opos, ovel);

        EXPECT_EQ(t1, t);
        compareTriplet(pos, opos);
        compareTriplet(vel, ovel);
    }
    fails += ::testing::Test::HasFailure();

}


int main(int argc, char **argv) {
    /*
     * Orbit unit-testing script.
     */
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();

}
