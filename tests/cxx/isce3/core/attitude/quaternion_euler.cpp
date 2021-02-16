// test suite for Quaternion class
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <isce3/core/DenseMatrix.h>
#include <isce3/core/EulerAngles.h>
#include <isce3/core/Quaternion.h>
#include <isce3/core/Vector.h>

using namespace isce3::core;

struct QuatEulerTest : public ::testing::Test {

    void SetUp() override
    {

        // clang-format off
    sc_pos << 
      -2434573.80388191110,
      -4820642.06528653484,
      4646722.94036952127;

    sc_vel << 
      522.99592536068,
      5107.80853161647,
      5558.15620986960;

    rot_mat << 
      0.0        ,  0.99987663, -0.01570732,
      -0.79863551, -0.0094529 , -0.60174078,
      -0.60181502,  0.01254442,  0.79853698;      

    quat_ant2ecf << 
      0.14889715185,
      0.02930644114,
      -0.90605724862,
      -0.39500763650;  

    ypr_ant2tcn <<
      -90.06934003*d2r,
      0.78478177*d2r,
      36.99994432*d2r;

        // clang-format on
    }

    // common vars
    const double d2r {M_PI / 180.0};
    const double r2d {1.0 / d2r};
    const double abs_err {1e-10};
    const double yaw {-0.9 * d2r}, pitch {0.06 * d2r}, roll {0.15 * d2r};
    const double mb_ang_deg {37.0};
    const double squint_ang_deg {-0.9};
    Vec3 sc_pos, sc_vel;
    Mat3 rot_mat;
    Vec4 quat_ant2ecf;
    Vec3 ypr_ant2tcn;
};

TEST_F(QuatEulerTest, EulerBasicConstruct)
{
    auto elr = EulerAngles(yaw, pitch, roll);
    ASSERT_NEAR(elr.yaw(), yaw, abs_err) << "Wrong Yaw for Euler obj";
    ASSERT_NEAR(elr.pitch(), pitch, abs_err) << "Wrong Pitch for Euler obj";
    ASSERT_NEAR(elr.roll(), roll, abs_err) << "Wrong Roll for Euler obj";
}

TEST_F(QuatEulerTest, QuatConstructMethod)
{
    //// Constructors

    // from non unity quaternion vector  Vec4
    auto uq_v4 = Quaternion(Vec4(2.0 * quat_ant2ecf));
    EXPECT_NEAR(uq_v4.norm(), 1.0, abs_err)
            << "Quat from Vec4 is not normalied!";
    EXPECT_NEAR((quat_ant2ecf.tail(3) - uq_v4.vec()).norm(), 0.0, abs_err)
            << "Imag/vec part of Quat from Vec4 is not correct!";
    EXPECT_NEAR(std::abs(quat_ant2ecf(0) - uq_v4.w()), 0.0, abs_err)
            << "Real/scalar part of Quat from Vec4 is not correct!";

    // from non-unity 3-D vector Vec3
    auto uq_v3 = Quaternion(sc_pos);
    EXPECT_NEAR(std::abs(uq_v3.w()), 0.0, abs_err)
            << "Real/scalar part of Quat from Vec3 is wrong";
    EXPECT_NEAR((uq_v3.vec() - sc_pos.normalized()).norm(), 0.0, abs_err)
            << "Imag/Vec part of Quat from Vec3 is wrong";

    // from unitary rotmat , Mat3
    auto uq_mat3 = Quaternion(rot_mat);
    EXPECT_NEAR((uq_mat3.toRotationMatrix() - rot_mat).cwiseAbs().maxCoeff(),
            0.0, 1e-8)
            << "Quat from rotation matrix and back fails!";

    // from YPR
    auto uq_yaw = Quaternion(Eigen::AngleAxisd(yaw, Vec3::UnitZ()));
    auto uq_pitch = Quaternion(Eigen::AngleAxisd(pitch, Vec3::UnitY()));
    auto uq_roll = Quaternion(Eigen::AngleAxisd(roll, Vec3::UnitX()));
    auto uq_ypr = Quaternion(yaw, pitch, roll);
    ASSERT_TRUE(uq_ypr.isApprox(uq_yaw * uq_pitch * uq_roll))
            << "Quat from YPR must be the same as AngleAxis products "
               "Yaw*Pitch*Roll";

    // from Euler object
    auto uq_elr = Quaternion(EulerAngles(yaw, pitch, roll));
    EXPECT_TRUE(uq_ypr.isApprox(uq_elr))
            << "Quat from YPR must be equal to Quat from EulerAngles";

    // from angle and 3-D vector
    auto uq_angaxis = Quaternion(yaw, Vec3(2.0 * Vec3::UnitZ()));
    EXPECT_TRUE(uq_angaxis.isApprox(uq_yaw))
            << "Quat from angle Yaw and scaled Z axis shall be Quat from "
               "AngleAxis for yaw";

    //// methods

    // to YPR
    auto ypr_vec = uq_ypr.toYPR();
    EXPECT_NEAR(ypr_vec(0), yaw, abs_err) << "Wrong yaw angle!";
    EXPECT_NEAR(ypr_vec(1), pitch, abs_err) << "Wrong pitch angle!";
    EXPECT_NEAR(ypr_vec(2), roll, abs_err) << "Wrong roll angle!";

    // to isce3 EulerAngle object
    auto elr_obj = uq_ypr.toEulerAngles();
    EXPECT_NEAR(elr_obj.yaw(), yaw, abs_err)
            << "Wrong Yaw angle for EulerAngles Obj";
    EXPECT_NEAR(elr_obj.pitch(), pitch, abs_err)
            << "Wrong Pitch angle for EulerAngles Obj";
    EXPECT_NEAR(elr_obj.roll(), roll, abs_err)
            << "Wrong Roll angle for EulerAngles Obj";

    // to Eigen AngleAxis object
    auto aa_obj = uq_angaxis.toAngleAxis();
    EXPECT_NEAR(std::abs(aa_obj.angle()), std::abs(yaw), abs_err)
            << "Angle must be +/-Yaw";
    EXPECT_NEAR(
            (aa_obj.axis().cwiseAbs() - Vec3::UnitZ()).cwiseAbs().maxCoeff(),
            0.0, abs_err)
            << "Axis must be +/-Z axis!";

    // a practical SAR example via Rotatation of Vec3 in ECEF
    auto uq_ant2ecf = Quaternion(quat_ant2ecf);
    auto ant_ecf = uq_ant2ecf.rotate(Vec3::UnitZ());
    Vec3 center_ecf {-sc_pos.normalized()};
    double mb_ang {r2d * std::acos(center_ecf.dot(ant_ecf))};
    EXPECT_NEAR(mb_ang, mb_ang_deg, 1e-1) << "Wrong Geocentric MB angle!";
    double squint_ang {r2d * std::asin(ant_ecf.dot(sc_vel.normalized()))};
    EXPECT_NEAR(squint_ang, squint_ang_deg, 1e-2) << "Wrong Squint angle";
}

TEST_F(QuatEulerTest, EulerConstructMethod)
{
    //// Constructors
    const auto quat_ypr = Quaternion(yaw, pitch, roll);
    const Mat3 matrot {quat_ypr.toRotationMatrix()};

    // from rotation mat
    auto elr_rotmat = EulerAngles(matrot);
    EXPECT_NEAR(elr_rotmat.yaw(), yaw, abs_err)
            << "Wrong Euler yaw angle from rotmat";
    EXPECT_NEAR(elr_rotmat.pitch(), pitch, abs_err)
            << "Wrong Euler pitch angle from rotmat";
    EXPECT_NEAR(elr_rotmat.roll(), roll, abs_err)
            << "Wrong Euler roll angle from rotmat";

    // from quaternion
    auto elr_quat = EulerAngles(quat_ypr);
    EXPECT_NEAR(elr_quat.yaw(), yaw, abs_err)
            << "Wrong Euler yaw angle from quat";
    EXPECT_NEAR(elr_quat.pitch(), pitch, abs_err)
            << "Wrong Euler pitch angle from quat";
    EXPECT_NEAR(elr_quat.roll(), roll, abs_err)
            << "Wrong Euler roll angle from quat";

    //// Methods

    // toRotationMatrix
    auto rotm = elr_quat.toRotationMatrix();
    EXPECT_NEAR((rotm - matrot).cwiseAbs().maxCoeff(), 0.0, abs_err)
            << "Wrong rotmat from Euler object!";

    // isApprox
    auto elr_other = EulerAngles(
            elr_quat.yaw() + abs_err, elr_quat.pitch(), elr_quat.roll());
    EXPECT_FALSE(elr_other.isApprox(elr_quat, abs_err))
            << "Two Euler angles must not be equal!";
    EXPECT_TRUE(elr_other.isApprox(elr_quat))
            << "Two Euler angles must be equal!";

    // rotate
    auto elr_ant2tcn =
            EulerAngles(ypr_ant2tcn(0), ypr_ant2tcn(1), ypr_ant2tcn(2));
    auto ant_tcn = elr_ant2tcn.rotate(Vec3::UnitZ());
    EXPECT_NEAR(r2d * std::acos(ant_tcn(2)), mb_ang_deg, 1e-2)
            << "Wrong Geodetic MB angle for Euler rotate!";

    // toQuaternion
    auto quat_elr = elr_quat.toQuaternion();
    EXPECT_TRUE(quat_elr.isApprox(quat_ypr, abs_err))
            << "Back and forth conversion between Euler and Quat fails";

    // in-place addition/subtraction
    auto elr_copy = EulerAngles(0.0, 0.0, 0.0);
    elr_copy += elr_quat;
    EXPECT_NEAR(elr_copy.yaw(), yaw, abs_err)
            << "Wrong Euler yaw angle after in-place add";
    EXPECT_NEAR(elr_copy.pitch(), pitch, abs_err)
            << "Wrong Euler pitch angle after in-place add";
    EXPECT_NEAR(elr_copy.roll(), roll, abs_err)
            << "Wrong Euler roll angle after in-place add";

    elr_copy -= elr_quat;
    EXPECT_NEAR(elr_copy.yaw(), 0.0, abs_err)
            << "Wrong Euler yaw angle after in-place sub";
    EXPECT_NEAR(elr_copy.pitch(), 0.0, abs_err)
            << "Wrong Euler pitch angle after in-place sub";
    EXPECT_NEAR(elr_copy.roll(), 0.0, abs_err)
            << "Wrong Euler roll angle after in-place sub";

    // in-place multiplication/concatenation
    elr_copy *= elr_quat;
    EXPECT_NEAR(elr_copy.yaw(), yaw, abs_err)
            << "Wrong Euler yaw angle after in-place mul";
    EXPECT_NEAR(elr_copy.pitch(), pitch, abs_err)
            << "Wrong Euler pitch angle after in-place mul";
    EXPECT_NEAR(elr_copy.roll(), roll, abs_err)
            << "Wrong Euler roll angle after in-place mul";

    // binary add/subtract operators
    auto elr_add = elr_quat + elr_quat;
    EXPECT_NEAR(elr_add.yaw(), 2.0 * yaw, abs_err)
            << "Wrong Euler yaw angle after binary add";
    EXPECT_NEAR(elr_add.pitch(), 2.0 * pitch, abs_err)
            << "Wrong Euler pitch angle after binary add";
    EXPECT_NEAR(elr_add.roll(), 2.0 * roll, abs_err)
            << "Wrong Euler roll angle after binary add";

    elr_add = elr_quat - elr_quat;
    EXPECT_NEAR(elr_add.yaw(), 0.0, abs_err)
            << "Wrong Euler yaw angle after binary sub";
    EXPECT_NEAR(elr_add.pitch(), 0.0, abs_err)
            << "Wrong Euler pitch angle after binary sub";
    EXPECT_NEAR(elr_add.roll(), 0.0, abs_err)
            << "Wrong Euler roll angle after binary sub";

    // binary multiplication/concatenation
    // For simplicity, an approximation is used for validation based on small
    // Euler angles (< 4.0 deg)
    auto elr_mul = (elr_quat + elr_quat) * elr_quat;
    EXPECT_NEAR(elr_mul.yaw(), 3.0 * yaw, 1e-4)
            << "Wrong Euler yaw angle after binary mul";
    EXPECT_NEAR(elr_mul.pitch(), 3.0 * pitch, 1e-4)
            << "Wrong Euler pitch angle after binary mul";
    EXPECT_NEAR(elr_mul.roll(), 3.0 * roll, 1e-4)
            << "Wrong Euler roll angle after binary mul";
}

int main(int argc, char** argv)
{

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
