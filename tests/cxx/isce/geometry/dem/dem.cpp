//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2018
//

#include <iostream>
#include <string>
#include <gtest/gtest.h>

// isce::core
#include <isce/core/Constants.h>

// isce::io
#include <isce/io/Raster.h>

// isce::geometry
#include <isce/geometry/DEMInterpolator.h>


TEST(DEMTest, ConstDEM) {

    //Create a constant height DEM
    float consthgt = 150.0;
    isce::geometry::DEMInterpolator dem(consthgt);

    //Check for initialization
    EXPECT_NEAR(dem.refHeight(), consthgt, 1.0e-6);
}

TEST(DEMTest, MethodConstruct) {

    //Constant height
    float consthgt = 220.0;

    //Methods to iterate over
    std::vector<isce::core::dataInterpMethod> methods = { isce::core::SINC_METHOD,
                                                          isce::core::BILINEAR_METHOD,
                                                          isce::core::BICUBIC_METHOD,
                                                          isce::core::NEAREST_METHOD,
                                                          isce::core::BIQUINTIC_METHOD };

    for(auto &method: methods)
    {
        isce::geometry::DEMInterpolator dem(consthgt, method);
        EXPECT_NEAR(dem.refHeight(), consthgt, 1.0e-6);
        EXPECT_EQ(dem.interpMethod(), method);
    }
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
