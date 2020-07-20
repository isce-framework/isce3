//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2018
//

#include <iostream>
#include <string>
#include <gtest/gtest.h>

// isce3::core
#include <isce3/core/Constants.h>

// isce3::io
#include <isce3/io/Raster.h>

// isce3::geometry
#include <isce3/geometry/DEMInterpolator.h>


TEST(DEMTest, ConstDEM) {

    //Create a constant height DEM
    float consthgt = 150.0;
    isce3::geometry::DEMInterpolator dem(consthgt);

    //Check for initialization
    EXPECT_NEAR(dem.refHeight(), consthgt, 1.0e-6);
}

TEST(DEMTest, MethodConstruct) {

    //Constant height
    float consthgt = 220.0;

    //Methods to iterate over
    std::vector<isce3::core::dataInterpMethod> methods = { isce3::core::SINC_METHOD,
                                                          isce3::core::BILINEAR_METHOD,
                                                          isce3::core::BICUBIC_METHOD,
                                                          isce3::core::NEAREST_METHOD,
                                                          isce3::core::BIQUINTIC_METHOD };

    for(auto &method: methods)
    {
        isce3::geometry::DEMInterpolator dem(consthgt, method);
        EXPECT_NEAR(dem.refHeight(), consthgt, 1.0e-6);
        EXPECT_EQ(dem.interpMethod(), method);
    }
}

int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
