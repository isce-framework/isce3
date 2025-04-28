#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <isce3/antenna/EdgeMethodCostFunc.h>
#include <isce3/core/Poly1d.h>
#include <isce3/math/polyfunc.h>

using namespace isce3::antenna;

// a helper function to evaluate a Poly1d object over Eigen array of inputs
Eigen::ArrayXd polyvals(const isce3::core::Poly1d& pf, Eigen::ArrayXd x)
{
    Eigen::Map<const Eigen::ArrayXd> pf_coef_map(
            pf.coeffs.data(), pf.coeffs.size());
    return isce3::math::polyval(pf_coef_map, x, pf.mean, pf.norm);
}

struct EdgeMethodCostFuncTest : public ::testing::Test {

    void SetUp() override
    {
        // form gain and look angle of reference antenna pattern
        // perform polyfiting to build Poly1d object version of reference
        // antenna pattern
        Eigen::Map<const Eigen::ArrayXd> lka_deg_map(
                lka_deg.data(), lka_deg.size());
        Eigen::Map<const Eigen::ArrayXd> gain_map(gain.data(), gain.size());
        pf_ref = isce3::math::polyfitObj(lka_deg_map * d2r, gain_map, 6, false);

        // uniformly-spaced look angles around rising edge used for both antenna
        // and echo objects
        Eigen::ArrayXd lka_edge_rad;
        lka_edge_rad = Eigen::ArrayXd::LinSpaced(
                num_lka_edge, d2r * min_lka_edge_deg, d2r * max_lka_edge_deg);

        // form ANT 3rd-order poly object with roll offset applied to edge look
        // angles
        pf_ant_vec.reserve(roll_ofs_ant_mdeg.size());
        for (const auto& roll : roll_ofs_ant_mdeg) {
            // add roll offset (perturbed)
            auto lka_ant_rad = lka_edge_rad + roll * md2r;
            // add a gain offset
            Eigen::ArrayXd gain_ant = polyvals(pf_ref, lka_ant_rad) + gain_ofs;
            pf_ant_vec.push_back(
                    isce3::math::polyfitObj(lka_edge_rad, gain_ant, 3, false));
        }

        // form echo 3rd-order poly object with roll offset applied to edge look
        // angles
        auto gain_echo = polyvals(pf_ref, lka_edge_rad);
        pf_echo = isce3::math::polyfitObj(lka_edge_rad, gain_echo, 3, false);

        // here we use constant but non-normalized weights (order 0)
        pf_wgt = isce3::core::Poly1d(0, 0.0, 1.0);
        pf_wgt.coeffs = std::vector<double> {10};
    }
    // List of methods
    void validate_estimation(const std::tuple<double, double, bool, int>& est,
            double roll_true_mdeg, const std::string& err_msg = {})
    {
        auto [roll_est, f_val, flag, n_iter] = est;
        // Absolute error (residual after compensating for estimated offset) in
        // (mdeg)
        auto abs_err = std::abs(roll_est * r2md + roll_true_mdeg);
        // check individual values
        std::string err_msg1 {"@ true roll offset " +
                              std::to_string(roll_true_mdeg) + " (mdeg) " +
                              err_msg};
        EXPECT_LE(n_iter, max_iter)
                << "Exceed max number of iteration " + err_msg1;
        EXPECT_TRUE(flag) << "Wrong convergence flag " + err_msg1;
        EXPECT_NEAR(abs_err, 0.0, max_abs_err_mdeg)
                << "Too large residual roll offset " + err_msg1;
        EXPECT_NEAR(f_val, 0.0, abs_tol)
                << "Wrong cost function value " + err_msg1;
    }

    // List of public members

    // conversion from (deg/mdeg) to (rad) and vice versa
    const double d2r {M_PI / 180.};
    const double md2r {d2r * 1e-3};
    const double r2md {1.0 / md2r};

    // max absolute pointing error (mdeg) of the estimation over wide range of
    // Roll angle offset. This is used to evaluate the residual error after
    // compensating for estimated offset. e.g. within [-200, +200] (mdeg, mdeg)
    // used here, it is set to around 1% margin of total 400 mdeg. This is way
    // finer than the requirement (>=15 mdeg)!
    const double max_abs_err_mdeg {5.0};

    // Absolute function value tolerance in root of cost function used in
    // "RollAngleOffsetFromEdge"
    const double abs_tol {1e-4};
    // Max expected iteration of cost function used in
    // "RollAngleOffsetFromEdge"
    const int max_iter {20};

    // look angle (off-nadir angle) inputs
    const double min_lka_edge_deg {32.8};
    const double max_lka_edge_deg {34.0};
    const double prec_lka_edge_deg {1e-3};
    const int num_lka_edge {
            static_cast<int>(std::round((max_lka_edge_deg - min_lka_edge_deg) /
                                        prec_lka_edge_deg) +
                             1)};

    // gain offset in (dB) between relative EL power patterns extracted from
    // antenna and echo. the roll offset estimation is insensitive to this gain
    // offset!
    const double gain_ofs {0.5};

    // desired roll angle offset in (mdeg) , ground truth values used for
    // validation. These value are also used to perturb EL power pattern from
    // antenna given the cost function tries to find a roll offset to be added
    // to antenna EL to align its power pattern with that of echo data. Thus,
    // the sign of estimated roll offset will be the opposite of these angles
    // with some tiny deviation due to poly fitting.
    std::vector<double> roll_ofs_ant_mdeg {-198.0, -42.5, 0.0, 67.0, 157.3};

    // Build a 6-order polyminals of a relative antenna gain from gain (dB)
    // versus look angles (rad) to be used as a reference for building both
    // antenna and echo data
    // These points are extracted from a realistic EL power pattern of ALOS1
    // beam #7.
    std::vector<double> gain {
            -2.2, -1.2, -0.55, -0.2, 0.0, -0.2, -0.5, -1.0, -2.0};
    std::vector<double> lka_deg {
            32.0, 32.5, 33.0, 33.5, 34.1, 34.5, 35., 35.5, 36.};
    isce3::core::Poly1d pf_ref;

    // a vector of 3rd-order polyfit objects for Antenna , one per each roll
    // angle offset (perturbed)
    std::vector<isce3::core::Poly1d> pf_ant_vec;

    // 3rd-order polyfit object for Echo common for all offset (unperturbed)
    isce3::core::Poly1d pf_echo;

    // Polyfit version of weights used in weighting cost function over look
    // angles (optional)
    isce3::core::Poly1d pf_wgt;
};

TEST_F(EdgeMethodCostFuncTest, RollAngleOffsetFromEdge_LookAngNearFar)
{
    // loop over roll offset (and antenna polyfit objects)
    for (std::size_t idx = 0; idx < roll_ofs_ant_mdeg.size(); ++idx) {
        // estimate roll offset w/o weighting
        auto est_tuple = rollAngleOffsetFromEdge(pf_echo, pf_ant_vec[idx],
                min_lka_edge_deg * d2r, max_lka_edge_deg * d2r,
                prec_lka_edge_deg * d2r);
        // validate results w/o weighting
        validate_estimation(est_tuple, roll_ofs_ant_mdeg[idx],
                std::string("w/o weighting!"));

        // estimate roll offset w/ weighting
        auto est_wgt_tuple = rollAngleOffsetFromEdge(pf_echo, pf_ant_vec[idx],
                min_lka_edge_deg * d2r, max_lka_edge_deg * d2r,
                prec_lka_edge_deg * d2r, pf_wgt);
        // validate results w/ weighting
        validate_estimation(est_wgt_tuple, roll_ofs_ant_mdeg[idx],
                std::string("w/ weighting!"));
    }
}

TEST_F(EdgeMethodCostFuncTest, RollAngleOffsetFromEdge_LookAngLinspace)
{
    // form isce3 Linspace object for uniformly sampled look angles within [min
    // ,max]
    auto lka_lsp = isce3::core::Linspace<double>::from_interval(
            min_lka_edge_deg * d2r, max_lka_edge_deg * d2r, num_lka_edge);
    // loop over roll offset (and antenna polyfit objects)
    for (std::size_t idx = 0; idx < roll_ofs_ant_mdeg.size(); ++idx) {
        // estimate roll offset w/o weighting
        auto est_tuple =
                rollAngleOffsetFromEdge(pf_echo, pf_ant_vec[idx], lka_lsp);
        // validate results
        validate_estimation(est_tuple, roll_ofs_ant_mdeg[idx],
                std::string("w/o weighting!"));

        // estimate roll offset w/ weighting
        auto est_wgt_tuple = rollAngleOffsetFromEdge(
                pf_echo, pf_ant_vec[idx], lka_lsp, pf_wgt);
        // validate results w/ weighting
        validate_estimation(est_wgt_tuple, roll_ofs_ant_mdeg[idx],
                std::string("w/ weighting!"));
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
