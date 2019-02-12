//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <gtest/gtest.h>

#include "isce/core/Utilities.h"
#include "isce/core/EulerAngles.h"


struct EulerTest : public ::testing::Test {

    typedef isce::core::EulerAngles EulerAngles;
    typedef isce::core::cartmat_t cartmat_t;

    double tol;
    cartmat_t R_ypr_ref, R_rpy_ref;
    EulerAngles attitude;

    protected:

        EulerTest() {

            // Make an array of epoch times
            std::vector<double> time = isce::core::linspace(0.0, 10.0, 20);

            // Make constant arrays of Euler angles
            std::vector<double> yaw, pitch, roll;
            for (size_t i = 0; i < time.size(); ++i) {
                yaw.push_back(0.1);
                pitch.push_back(0.05);
                roll.push_back(-0.1);
            }

            // Set data for EulerAngles object
            attitude = EulerAngles(time, yaw, pitch, roll);

            // Define the reference rotation matrix (YPR)
            R_ypr_ref = {{
                {0.993760669166, -0.104299329454, 0.039514330251},
                {0.099708650872, 0.989535160981, 0.104299329454},
                {-0.049979169271, -0.099708650872, 0.993760669166}
            }};

            // Define the reference rotation matrix (RPY)
            R_rpy_ref = {{
                {0.993760669166, -0.099708650872, 0.049979169271},
                {0.094370001341, 0.990531416861, 0.099708650872},
                {-0.059447752410, -0.094370001341, 0.993760669166}
            }};

            // Set tolerance
            tol = 1.0e-10;
        }

        ~EulerTest() {}
};

TEST_F(EulerTest, CheckYPR) {
    cartmat_t R_ypr = attitude.rotmat(5.0, "ypr");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            ASSERT_NEAR(R_ypr_ref[i][j], R_ypr[i][j], tol);
        }
    }
}

TEST_F(EulerTest, CheckRPY) {
    cartmat_t R_rpy = attitude.rotmat(5.0, "rpy");
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
