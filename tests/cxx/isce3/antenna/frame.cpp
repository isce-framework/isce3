// Test suite for Antenna::Frame class
#include <isce3/antenna/forward.h>

#include <cstddef>
#include <iostream>

#include <gtest/gtest.h>

#include <isce3/antenna/Frame.h>

using namespace isce3::antenna;

struct FrameTest : public ::testing::Test {

    void SetUp() override
    {
        // Convert all angles from degrees to radians
        std::for_each(
                el_deg.begin(), el_deg.end(), [=](double& ang) { ang *= d2r; });
        std::for_each(
                az_deg.begin(), az_deg.end(), [=](double& ang) { ang *= d2r; });
        // fill out a vector of expected XYZ values for each (EL,AZ) pair
        est_xyz_eaz.reserve(el_deg.size());
        est_xyz_eoa.reserve(el_deg.size());
        est_xyz_aoe.reserve(el_deg.size());
        est_xyz_tp.reserve(el_deg.size());

        est_xyz_eaz[0] << -0.121864326797, -0.015668270588, 0.992423090799;
        est_xyz_eaz[1] << -0.034899213187, -0.006979842637, 0.999366462673;
        est_xyz_eaz[2] << 0.000000000000, 0.000000000000, 1.000000000000;
        est_xyz_eaz[3] << 0.017452264667, 0.006980905867, 0.999823327099;
        est_xyz_eaz[4] << 0.069753604227, 0.015694560951, 0.997440782931;

        est_xyz_eoa[0] << -0.121854308687, -0.015707317312, 0.992423703686;
        est_xyz_eoa[1] << -0.034898646226, -0.006981260298, 0.999366472570;
        est_xyz_eoa[2] << 0.000000000000, 0.000000000000, 1.000000000000;
        est_xyz_eoa[3] << 0.017451981134, 0.006981260298, 0.999823329573;
        est_xyz_eoa[4] << 0.069747868061, 0.015707317312, 0.997440983259;

        est_xyz_aoe[0] << -0.121869343405, -0.015590237350, 0.992423703686;
        est_xyz_aoe[1] << -0.034899496703, -0.006977007503, 0.999366472570;
        est_xyz_aoe[2] << 0.000000000000, 0.000000000000, 1.000000000000;
        est_xyz_aoe[3] << 0.017452406437, 0.006980197018, 0.999823329573;
        est_xyz_aoe[4] << 0.069756473744, 0.015669055076, 0.997440983259;

        est_xyz_tp[0] << 0.121854308687, -0.001914240447, 0.992546151641;
        est_xyz_tp[1] << 0.034898646226, -0.000243642471, 0.999390827019;
        est_xyz_tp[2] << 0.000000000000, 0.000000000000, 1.000000000000;
        est_xyz_tp[3] << 0.017451981134, 0.000121839792, 0.999847695156;
        est_xyz_tp[4] << 0.069747868061, 0.001095687068, 0.997564050260;
    }

    // list of functions
    /**
     * @internal
     * Validate results for bilateral transform between (EL/THETA, AZ/PHI) and
     * (X,Y,Z)
     * @param[in] antfrm : antenna Frame object
     * @param[in] el_tht : a vector of EL or THETA angles in radians
     * @param[in] az_phi : a vector of AZ or PHI angle in radians
     * @param[in] est_xyz : a vector of estimated cartesian values in Eigen
     * Vector3d format
     */
    void validateResults(const Frame& antfrm, const std::vector<double>& el_tht,
            const std::vector<double>& az_phi,
            const std::vector<isce3::core::Vec3>& est_xyz)
    {
        std::string grid_str {toStr(antfrm.gridType())};
        for (std::size_t idx = 0; idx < el_tht.size(); ++idx) {
            // from spherical to cartesian
            auto v_xyz {antfrm.sphToCart(el_tht[idx], az_phi[idx])};
            EXPECT_NEAR(
                    (v_xyz - est_xyz[idx]).cwiseAbs().maxCoeff(), 0.0, abs_err)
                    << "Wrong XYZ vector from scalar (EL/THETA, AZ/PHI) = ("
                    << el_tht[idx] * r2d << ',' << az_phi[idx] * r2d
                    << ") in (deg,deg) for grid " << grid_str;

            // from cartesian to spherical
            auto v_elaz {antfrm.cartToSph(v_xyz)};
            EXPECT_NEAR(std::abs(v_elaz[0] - el_tht[idx]) * r2d, 0.0, abs_err)
                    << "Wrong EL angle from Vec3 cartesian to scalar "
                       "(EL/THETA, AZ/PHI) for grid "
                    << grid_str;
            EXPECT_NEAR(std::abs(v_elaz[1] - az_phi[idx]) * r2d, 0.0, abs_err)
                    << "Wrong AZ angle from Vec3 cartesian to scalar "
                       "(EL/THETA, AZ/PHI) for grid "
                    << grid_str;
        }
    }

    // list of common  vars
    const double d2r {M_PI / 180.0};
    const double r2d {1.0 / d2r};
    const double abs_err {1e-7};
    // EL/AZ or THETA/PHI angles all in (degrees)
    std::vector<double> el_deg {-7.0, -2.0, 0.0, 1.0, 4.0};
    std::vector<double> az_deg {-0.9, -0.4, 0.0, 0.4, 0.9};
    // Estimated XYZ values for each pair of (EL,AZ) in EL_AND_AZ
    std::vector<isce3::core::Vec3> est_xyz_eaz;
    // Estimated XYZ values for each pair of (THETA,PHI) in THETA_PHI
    std::vector<isce3::core::Vec3> est_xyz_tp;
    // Estimated XYZ values for each pair of (EL,AZ) in EL_OVER_AZ
    std::vector<isce3::core::Vec3> est_xyz_eoa;
    // Estimated XYZ values for each pair of (EL,AZ) in AZ_OVER_EL
    std::vector<isce3::core::Vec3> est_xyz_aoe;
};

TEST_F(FrameTest, ConstructGridTypeToStr)
{
    // default constructor
    auto antObj {Frame()};
    auto grdType {antObj.gridType()};
    EXPECT_EQ(antObj.gridType(), SphGridType::EL_AND_AZ)
            << "Default grid type must be enum EL_AND_AZ!";
    EXPECT_TRUE(toStr(grdType) == std::string {"EL_AND_AZ"})
            << "String equivalent of grid type must be a string 'EL_AND_AZ'";

    // enum constructor
    antObj = Frame(SphGridType::THETA_PHI);
    grdType = antObj.gridType();
    EXPECT_EQ(antObj.gridType(), SphGridType::THETA_PHI)
            << "Grid type must be enum THETA_PHI for enum constructor!";
    EXPECT_TRUE(toStr(grdType) == std::string {"THETA_PHI"})
            << "String equivalent of grid type must be a string 'THETA_PHI'";

    // string constructor
    antObj = Frame("EL_AND_AZ");
    grdType = antObj.gridType();
    EXPECT_EQ(antObj.gridType(), SphGridType::EL_AND_AZ)
            << "The grid type must be enum EL_AND_AZ for str constructor";
    EXPECT_TRUE(toStr(grdType) == std::string {"EL_AND_AZ"})
            << "String equivalent of grid for str constructor shall be "
               "'EL_AND_AZ'";
    antObj = Frame("theta_phi");
    grdType = antObj.gridType();
    EXPECT_EQ(antObj.gridType(), SphGridType::THETA_PHI)
            << "The grid type must be enum THETA_PHI for str constructor";
    EXPECT_TRUE(toStr(grdType) == std::string {"THETA_PHI"})
            << "String equivalent of grid for str constructor shall be "
               "'THETA_PHI'";

    // throw exception for bad string
    EXPECT_THROW(Frame("EL_AZ"), isce3::except::InvalidArgument)
            << "Must throw isce3 exception InvalidArgument for bad string";
}

TEST_F(FrameTest, ScalarMethods)
{
    // EL_AND_AZ Grid  (Default)
    validateResults(Frame(), el_deg, az_deg, est_xyz_eaz);

    // THETA_PHI Grid
    auto tht_ang = el_deg;
    std::for_each(tht_ang.begin(), tht_ang.end(),
            [](double& ang) { ang = std::abs(ang); });
    validateResults(Frame(SphGridType::THETA_PHI), tht_ang, az_deg, est_xyz_tp);

    // EL_OVER_AZ Grid
    validateResults(
            Frame(SphGridType::EL_OVER_AZ), el_deg, az_deg, est_xyz_eoa);

    // AZ_OVER_EL Grid
    validateResults(
            Frame(SphGridType::AZ_OVER_EL), el_deg, az_deg, est_xyz_aoe);
}

TEST_F(FrameTest, VectorMethods)
{
    auto antfrm {Frame()};
    // from spherical to cartesian
    auto v_xyz {antfrm.sphToCart(el_deg, az_deg)};
    for (std::size_t idx = 0; idx < v_xyz.size(); ++idx)
        EXPECT_NEAR((v_xyz[idx] - est_xyz_eaz[idx]).cwiseAbs().maxCoeff(), 0.0,
                abs_err)
                << "Wrong XYZ vector from vector (EL,AZ) = ("
                << el_deg[idx] * r2d << ',' << az_deg[idx] * r2d
                << ") in (deg,deg)";
    // from cartesian to spherical
    auto v_elaz {antfrm.cartToSph(v_xyz)};
    for (std::size_t idx = 0; idx < v_xyz.size(); ++idx) {
        EXPECT_NEAR(std::abs(v_elaz[idx](0) - el_deg[idx]) * r2d, 0.0, abs_err)
                << "Wrong EL angle from vector of Vec3 cartesian to vector of "
                   "Vec2 EL/AZ";
        EXPECT_NEAR(std::abs(v_elaz[idx](1) - az_deg[idx]) * r2d, 0.0, abs_err)
                << "Wrong AZ angle from vector of Vec3 cartesian to vector of "
                   "Vec2 EL/AZ";
    }
}

TEST_F(FrameTest, MixedVecScalMethods)
{
    auto antfrm {Frame()};
    // EL cut at a fixed AZ angle
    auto v_elaz {antfrm.cartToSph(antfrm.sphToCart(el_deg, az_deg[0]))};
    for (std::size_t idx = 0; idx < v_elaz.size(); ++idx) {
        EXPECT_NEAR(std::abs(v_elaz[idx](0) - el_deg[idx]) * r2d, 0.0, abs_err)
                << "Wrong EL angle from vector EL and scalar AZ to Cart and "
                   "back";
        EXPECT_NEAR(std::abs(v_elaz[idx](1) - az_deg[0]) * r2d, 0.0, abs_err)
                << "Wrong AZ angle from vector EL and scalar AZ to Cart and "
                   "back";
    }
    // AZ cut at a fixed EL angle
    v_elaz = antfrm.cartToSph(antfrm.sphToCart(el_deg[0], az_deg));
    for (std::size_t idx = 0; idx < v_elaz.size(); ++idx) {
        EXPECT_NEAR(std::abs(v_elaz[idx](0) - el_deg[0]) * r2d, 0.0, abs_err)
                << "Wrong EL angle from scalar  EL and vector AZ to Cart and "
                   "back";
        EXPECT_NEAR(std::abs(v_elaz[idx](1) - az_deg[idx]) * r2d, 0.0, abs_err)
                << "Wrong AZ angle from scalar EL and vector AZ to Cart and "
                   "back";
    }
}

TEST_F(FrameTest, OperatorsEq)
{
    // assignment
    auto antObj {Frame()};
    EXPECT_EQ(antObj.gridType(), SphGridType::EL_AND_AZ)
            << "Expected a default grid type before assignment";
    antObj = Frame("EL_OVER_AZ");
    EXPECT_EQ(antObj.gridType(), SphGridType::EL_OVER_AZ)
            << "Expected a new grid type after assignment";

    // equality
    EXPECT_TRUE(Frame() == Frame("el_and_az"))
            << "Expect equality between two constructors";

    // inequality
    EXPECT_TRUE(Frame() != Frame("theta_phi"))
            << "Expect inequality between two constructors";
}

int main(int argc, char** argv)
{

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
