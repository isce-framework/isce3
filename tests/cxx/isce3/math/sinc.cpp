#include <cmath>
#include <gtest/gtest.h>
#include <isce3/math/Sinc.h>

using isce3::math::sinc;

TEST(SincTest, Sinc)
{
    double errtol = 1e-16;
    EXPECT_NEAR(sinc(0.),   1.,        errtol);
    EXPECT_LT(sinc(1e-6),   1.);
    EXPECT_NEAR(sinc(1.),   0.,        errtol);
    EXPECT_NEAR(sinc(0.5),  2./M_PI,   errtol);
    EXPECT_NEAR(sinc(-1.3), sinc(1.3), errtol);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
