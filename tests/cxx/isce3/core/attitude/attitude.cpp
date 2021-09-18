#include "isce3/core/Attitude.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <isce3/core/DenseMatrix.h>

#include "isce3/core/EulerAngles.h"
#include "isce3/core/Utilities.h"

using namespace isce3::core;

struct AttitudeTest : public ::testing::Test {

    double t0, tf, tol;
    cartmat_t R_ypr_ref, R_rpy_ref;
    Attitude attitude;
    DateTime epoch;

protected:
    AttitudeTest()
    {
        // Make an array of epoch times
        t0 = 0.0;
        tf = 10.0;
        std::vector<double> time = linspace(t0, tf, 21);

        // Make constant array of quaternions.
        EulerAngles ypr(0.1, 0.05, -0.1);
        std::vector<Quaternion> quaternions(time.size(), Quaternion(ypr));

        // Set data for Attitude object
        epoch = DateTime(2020, 1, 1);
        attitude = Attitude(time, quaternions, epoch);

        // Define the reference rotation matrix (YPR)
        R_ypr_ref = cartmat_t{{{0.993760669166, -0.104299329454, 0.039514330251},
                               {0.099708650872, 0.989535160981, 0.104299329454},
                               {-0.049979169271, -0.099708650872, 0.993760669166}}};

        // Set tolerance
        tol = 1.0e-10;
    }

    ~AttitudeTest() {}
};

TEST_F(AttitudeTest, Interpolate)
{
    Mat3 R_ypr = attitude.interpolate(5.0).toRotationMatrix();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            ASSERT_NEAR(R_ypr_ref(i, j), R_ypr(i, j), tol);
        }
    }
}

TEST_F(AttitudeTest, Time)
{
    EXPECT_DOUBLE_EQ(t0, attitude.startTime());
    EXPECT_DOUBLE_EQ(tf, attitude.endTime());
    EXPECT_EQ(epoch + TimeDelta(t0), attitude.startDateTime());
    EXPECT_EQ(epoch + TimeDelta(tf), attitude.endDateTime());
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
