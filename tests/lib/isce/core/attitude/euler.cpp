//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <cstdio>
#include <cmath>
#include <gtest/gtest.h>

#include "isce/core/EulerAngles.h"


struct EulerTest : public ::testing::Test {

    typedef isce::core::EulerAngles EulerAngles;

    double yaw, pitch, roll, tol;
    std::vector<std::vector<double>> R_ypr_ref, R_rpy_ref;
    EulerAngles attitude;

    protected:

        EulerTest() {

            // Set the attitude angles
            yaw = 0.1;
            pitch = 0.05;
            roll = -0.1;

            // Instantate attitude object
            attitude = EulerAngles(yaw, pitch, roll);

            // Define the reference rotation matrix (YPR)
            R_ypr_ref = {
                {0.993760669166, -0.104299329454, 0.039514330251},
                {0.099708650872, 0.989535160981, 0.104299329454},
                {-0.049979169271, -0.099708650872, 0.993760669166}
            };

            // Define the reference rotation matrix (RPY)
            R_rpy_ref = {
                {0.993760669166, -0.099708650872, 0.049979169271},
                {0.094370001341, 0.990531416861, 0.099708650872},
                {-0.059447752410, -0.094370001341, 0.993760669166}
            };

            // Set tolerance
            tol = 1.0e-10;
        }

        ~EulerTest() {
            R_ypr_ref.clear();
            R_rpy_ref.clear();
        }
};

TEST_F(EulerTest, CheckYPR) {
    std::vector<std::vector<double>> R_ypr = attitude.rotmat("ypr");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            ASSERT_NEAR(R_ypr_ref[i][j], R_ypr[i][j], tol);
        }
    }
}

TEST_F(EulerTest, CheckRPY) {
    std::vector<std::vector<double>> R_rpy = attitude.rotmat("rpy");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            ASSERT_NEAR(R_rpy_ref[i][j], R_rpy[i][j], tol);
        }
    }
}


int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
