#include <iostream>

#include <gtest/gtest.h>

#include <isce3/core/Basis.h>
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/EulerAngles.h>
#include <isce3/core/Quaternion.h>
#include <isce3/core/Vector.h>

using namespace isce3::core;

TEST(EulerAngles, RoundTrip)
{
    const double tol = 1e-12;
    const auto ypr = EulerAngles(0.1, 0.05, -0.1);
    auto rt = EulerAngles(ypr.toRotationMatrix());
    EXPECT_NEAR(rt.yaw(), ypr.yaw(), tol);
    EXPECT_NEAR(rt.pitch(), ypr.pitch(), tol);
    EXPECT_NEAR(rt.roll(), ypr.roll(), tol);

    rt = EulerAngles(Quaternion(ypr));
    EXPECT_NEAR(rt.yaw(), ypr.yaw(), tol);
    EXPECT_NEAR(rt.pitch(), ypr.pitch(), tol);
    EXPECT_NEAR(rt.roll(), ypr.roll(), tol);
}

TEST(Quaternion, RoundTrip)
{
    const double tol = 1e-12;
    const auto q = Quaternion(1, 2, 3, 4); // normalized
    auto rt = Quaternion(q.toRotationMatrix());
    EXPECT_NEAR(rt.w(), q.w(), tol);
    EXPECT_NEAR(rt.x(), q.x(), tol);
    EXPECT_NEAR(rt.y(), q.y(), tol);
    EXPECT_NEAR(rt.z(), q.z(), tol);

    rt = Quaternion(EulerAngles(q));
    EXPECT_NEAR(rt.w(), q.w(), tol);
    EXPECT_NEAR(rt.x(), q.x(), tol);
    EXPECT_NEAR(rt.y(), q.y(), tol);
    EXPECT_NEAR(rt.z(), q.z(), tol);
}

TEST(Quaternion, Access)
{
    // Super pedantic because Eigen storage order is not wxyz!
    auto q = Quaternion(1, 0, 0, 0);
    EXPECT_EQ(q.w(), 1.0);
    EXPECT_EQ(q.x(), 0.0);
    EXPECT_EQ(q.y(), 0.0);
    EXPECT_EQ(q.z(), 0.0);
    // See!
    EXPECT_NE(q.coeffs()(0), q.w());

    auto qv = Vector<4, double>(1, 2, 3, 4);
    qv /= qv.norm();
    q = Quaternion(qv(0), qv(1), qv(2), qv(3));
    // May do normalization differently, so allow some difference.
    EXPECT_DOUBLE_EQ(q.w(), qv(0));
    EXPECT_DOUBLE_EQ(q.x(), qv(1));
    EXPECT_DOUBLE_EQ(q.y(), qv(2));
    EXPECT_DOUBLE_EQ(q.z(), qv(3));
}

TEST(Quaternion, Norm)
{
    // Quaternion should always be normalized.
    auto q = Quaternion(2, 0, 0, 0);
    EXPECT_DOUBLE_EQ(q.w(), 1.0);
}

TEST(EulerAngles, Access)
{
    const auto ypr = EulerAngles(1, 2, 3);
    EXPECT_EQ(ypr.yaw(), 1.0);
    EXPECT_EQ(ypr.pitch(), 2.0);
    EXPECT_EQ(ypr.roll(), 3.0);
}

void ExpectRotmatNear(const Mat3& a, const Mat3& b, double tol)
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(a(i, j), b(i, j), tol);
        }
    }
}

TEST(EulerAngles, Values)
{
    double tol = 1e-12;
    auto ypr = EulerAngles(0, 0, 0);
    Mat3 expected = Mat3::Identity();
    ExpectRotmatNear(ypr.toRotationMatrix(), expected, tol);

    ypr = EulerAngles(M_PI, 0, 0);
    // clang-format off
    expected << -1,  0,  0,
                 0, -1,  0,
                 0,  0,  1;
    ExpectRotmatNear(ypr.toRotationMatrix(), expected, tol);

    ypr = EulerAngles(0, M_PI/2, 0);
    expected <<  0,  0,  1,
                 0,  1,  0,
                -1,  0,  0;
    ExpectRotmatNear(ypr.toRotationMatrix(), expected, tol);

    ypr = EulerAngles(0, 0, M_PI);
    expected <<  1,  0,  0,
                 0, -1,  0,
                 0,  0, -1;
    ExpectRotmatNear(ypr.toRotationMatrix(), expected, tol);

    // Test from old code, should catch error in order of YPR sequence.
    // Unknown origin of "golden" matrix.
    ypr = EulerAngles(0.1, 0.05, -0.1);
    expected << 0.993760669166, -0.104299329454, 0.039514330251,
                0.099708650872,  0.989535160981, 0.104299329454,
               -0.049979169271, -0.099708650872, 0.993760669166;
    ExpectRotmatNear(ypr.toRotationMatrix(), expected, tol);
    // clang-format on
}

TEST(Quaternion, Values)
{
    auto q = Quaternion(1, 0, 0, 0);
    ExpectRotmatNear(q.toRotationMatrix(), Mat3::Identity(), 1e-12);
}

static inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

TEST(Basis, FactoredYPR)
{
    // This pair is from NISAR Radar Echo Emulator (REE) "out17" test case.
    auto angles = EulerAngles(deg2rad(-3.8946948276198485),
                              deg2rad(8.604948585720144e-16), deg2rad(0.0));
    auto quaternion = Quaternion(-0.5641527828808183, 0.29095772599574443,
                                 0.6570552008158168, 0.40663706463147226);
    // Corresponding position and velocity.
    Vec3 position {3595770.2903752117, -6145673.956536981, 233357.30879290705};
    Vec3 velocity {-1469.6807450495169, -578.7892748024412, 7403.171539154211};

    // REE used geocentric TCN to generate this pair.
    auto out = factoredYawPitchRoll(quaternion, position, velocity);
    double tol = 1e-8;
    EXPECT_NEAR(out.yaw(), angles.yaw(), tol);
    EXPECT_NEAR(out.pitch(), angles.pitch(), tol);
    EXPECT_NEAR(out.roll(), angles.roll(), tol);

    // Check vs geodetic TCN with loose tolerance just to make sure we didn't
    // accidentally transpose or something.
    auto ellipsoid = Ellipsoid();
    out = factoredYawPitchRoll(quaternion, position, velocity, ellipsoid);
    tol = 1e-2;
    EXPECT_NEAR(out.yaw(), angles.yaw(), tol);
    EXPECT_NEAR(out.pitch(), angles.pitch(), tol);
    EXPECT_NEAR(out.roll(), angles.roll(), tol);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
