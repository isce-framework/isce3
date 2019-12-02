//-*- C++ -*-
//-*- coding: utf-8 -*-


#include <iostream>
#include <cmath>
#include <vector>
#include "isce/core/Poly2d.h"
#include "gtest/gtest.h"


struct Poly2dTest : public ::testing::Test {
    virtual void SetUp() {
        fails = 0;
    }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "Poly2d::TearDown sees failures" << std::endl;
        }
    }
    unsigned fails;
};


TEST_F(Poly2dTest, Constant) {

    const double refval = 10.0;

    // Interpolate N values in x and y
    for (size_t i = 1; i < 5; ++i)
    {
        //Mean and norm should not matter
        isce::core::Poly2d poly(0, 0, i*1.0, 0, i*i*1.0, 1.0);
        poly.setCoeff(0, 0, refval);

        double value = poly.eval(0.0, i*1.0);
        EXPECT_DOUBLE_EQ(value, refval);
    }
    
    fails += ::testing::Test::HasFailure();
}


TEST_F(Poly2dTest, MeanShift)
{
    //Use identity polynomial for testing
    isce::core::Poly2d refpoly(2, 0, 0.0, 0.0, 1.0, 1.0);
    refpoly.setCoeff(0, 0, 0.0);
    refpoly.setCoeff(0, 1, 1.0);
    refpoly.setCoeff(0, 2, 0.0);

    for(size_t i=0; i<5; i++)
    {
        isce::core::Poly2d newpoly(refpoly);
        newpoly.rangeMean = 0.5 * i * i;

        double refval = refpoly.eval(0.0, 2.0 * i);
        double newval = newpoly.eval(0.0, 2.0 * i + 0.5 * i * i);
        EXPECT_DOUBLE_EQ(newval, refval);
    }

    fails += ::testing::Test::HasFailure();
}


TEST_F(Poly2dTest, NormShift)
{
    //Use square polynomial for testing
    isce::core::Poly2d refpoly(2, 0, 0.0, 0.0, 1.0, 1.0);
    refpoly.setCoeff(0, 0, 0.0);
    refpoly.setCoeff(0, 1, 0.0);
    refpoly.setCoeff(0, 2, 1.0);

    for(size_t i=1; i<6; i++)
    {
        isce::core::Poly2d newpoly(refpoly);
        newpoly.rangeNorm = i * i * 1.0;

        double refval = refpoly.eval(0.0, 2.5);
        double newval = newpoly.eval(0.0, 2.5 * i * i);

        EXPECT_DOUBLE_EQ(newval, refval);
    }

    fails += ::testing::Test::HasFailure();
}

int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

// end of file
