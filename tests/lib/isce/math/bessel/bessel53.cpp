//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <iostream>
#include <isce/math/Bessel.h>
#include <gtest/gtest.h>

struct BesselTest : public ::testing::Test {};


#define besselTest(name,x,r)       \
    TEST_F(BesselTest, name) {     \
        double y = isce::math::bessel_i0(x); \
        EXPECT_NEAR(y, r, 1.0e-12);\
    } struct consume_semicolon


besselTest(zero, 0.0, 1.0);
besselTest(one, 1.0, 1.26606587775200833559824462521471753760767031135496220680814);
besselTest(minus1, -1.0, 1.26606587775200833559824462521471753760767031135496220680814);
besselTest(minus2, -2.0, 2.27958530233606726743720444081153335328584110278545905407084);
besselTest(four, 4.0, 11.3019219521363304963562701832171024974126165944353377060065);
besselTest(minus7, -7.0, 168.593908510289698857326627187500840376522679234531714193194);
besselTest(random, 0.0009765625, 1.00000023841859331241759166109699567801556273303717896447683);
besselTest(small, 9.5367431640625e-7, 1.00000000000022737367544324498417583090700894607432256476338);

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
