//-*- C++ -*-
//-*- coding: utf-8 -*-


#include <iostream>
#include <cmath>
#include <vector>
#include "isce/core/Poly1d.h"
#include "gtest/gtest.h"


struct Poly1DTest : public ::testing::Test {
    virtual void SetUp() {
        fails = 0;
    }
    virtual void TearDown() {
        if (fails > 0) {
            std::cerr << "Poly1D::TearDown sees failures" << std::endl;
        }
    }
    unsigned fails;
};


TEST_F(Poly1DTest, Constant) {

    const double refval = 10.0;

    // Interpolate N values in x and y
    for (size_t i = 1; i < 5; ++i)
    {
        //Mean and norm should not matter
        isce::core::Poly1d poly(0, i*1.0, i*i*1.0);
        poly.setCoeff(0, refval);

        double value = poly.eval(i*1.0);
        EXPECT_DOUBLE_EQ(value, refval);
    }
    
    fails += ::testing::Test::HasFailure();
}


TEST_F(Poly1DTest, MeanShift)
{
    //Use identity polynomial for testing
    isce::core::Poly1d refpoly(2, 0.0, 1.0);
    refpoly.setCoeff(0, 0.0);
    refpoly.setCoeff(1, 1.0);
    refpoly.setCoeff(2, 0.0);

    for(size_t i=0; i<5; i++)
    {
        isce::core::Poly1d newpoly(refpoly);
        newpoly.mean = 0.5 * i * i;

        double refval = refpoly.eval(2.0 * i);
        double newval = newpoly.eval(2.0 * i + 0.5 * i * i);
        EXPECT_DOUBLE_EQ(newval, refval);
    }

    fails += ::testing::Test::HasFailure();
}


TEST_F(Poly1DTest, NormShift)
{
    //Use square polynomial for testing
    isce::core::Poly1d refpoly(2,0.0,1.0);
    refpoly.setCoeff(0, 0.0);
    refpoly.setCoeff(1, 0.0);
    refpoly.setCoeff(2, 1.0);

    for(size_t i=1; i<6; i++)
    {
        isce::core::Poly1d newpoly(refpoly);
        newpoly.norm = i * i * 1.0;

        double refval = refpoly.eval(2.5);
        double newval = newpoly.eval(2.5 * i * i);

        EXPECT_DOUBLE_EQ(newval, refval);
    }

    fails += ::testing::Test::HasFailure();
}

TEST_F(Poly1DTest, Derivative)
{
    for(size_t i=1; i<6; i++)
    {
        isce::core::Poly1d refpoly(i, 0.0, 1.0);
        refpoly.norm = i;
        refpoly.setCoeff(0, 10.0);
        for (int ii=1; ii<=static_cast<int>(i); ii++)
        {
            refpoly.setCoeff(ii, 1.0/ii);
        }


        isce::core::Poly1d refder(i-1,0.0,1.0);
        refder.norm = i;
        std::fill( refder.coeffs.begin(), refder.coeffs.end(), 1.0/i);

        isce::core::Poly1d newpoly = refpoly.derivative();

        double refval = refder.eval(0.8);
        double newval = newpoly.eval(0.8);

        EXPECT_DOUBLE_EQ(newval, refval);
    }

    fails += ::testing::Test::HasFailure();
}


int main(int argc, char **argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

// end of file
