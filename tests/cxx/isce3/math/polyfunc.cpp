#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <isce3/math/polyfunc.h>

using namespace isce3::math;

struct PolyfuncTest : public ::testing::Test {

    void SetUp() override
    {
        // initialize the members
        y_vals = get_y_vals(x_vals, pn_coef_3rd);

        // calculate std and mean of the input x_vals
        Eigen::Map<const Eigen::ArrayXd> x_vals_m(x_vals.data(), x_vals.size());
        x_mean = x_vals_m.mean();
        x_std = cal_std(x_vals_m);

        // centralize/scale the input and get the new centralized coeff "coef_c"
        Eigen::ArrayXd x_vals_c(x_vals_m);
        x_vals_c -= x_vals_c.mean();
        x_vals_c /= cal_std(x_vals_c);
        double mean_c, std_c;
        std::tie(coef_c, mean_c, std_c) = polyfit(x_vals_c, y_vals, 3);
    }

    // list of functions
    Eigen::ArrayXd get_y_vals(
            const std::vector<double>& x, const std::vector<double>& coef3rd)
    {
        // c0 + c1 * x + c2 * x^2 + c3 * x^3
        Eigen::Map<const Eigen::ArrayXd> x_m(x.data(), x.size());
        Eigen::Map<const Eigen::ArrayXd> coef_m(coef3rd.data(), coef3rd.size());
        return coef_m(0) + coef_m(1) * x_m + coef_m(2) * x_m.pow(2) +
               coef_m(3) * x_m.pow(3);
    }

    void validate_coeffs(const Eigen::Ref<const Eigen::ArrayXd>& coef_est,
            const Eigen::Ref<const Eigen::ArrayXd>& coef_true,
            const std::string& message = {})
    {
        ASSERT_EQ(coef_est.size(), coef_true.size())
                << "Size mismtach between estimated and true coeffs " + message;
        EXPECT_NEAR((coef_est - coef_true).abs().maxCoeff(), 0.0, abs_tol)
                << "Value mismatch between estimated and true coeffs " +
                           message;
    }

    double cal_std(const Eigen::Ref<const Eigen::ArrayXd>& x)
    {
        return std::sqrt((x - x.mean()).abs2().mean());
    }

    // list of members
    const double abs_tol {1e-7};
    // 3rd order poly nomial coeffs in ascending order
    std::vector<double> pn_coef_3rd {1, 2, -3, 4};
    // 2ed polynomial which is derivative of 3rd order one above in ascending
    // order
    std::vector<double> pn_coef_2ed {2, -6, 12};
    // input x data with size >= 4!
    std::vector<double> x_vals {-4.5, -3, -0.2, 0.5, 3.4, 6, 6.5};
    // polynomial evaluated y values from x values via 3rd polynomial
    Eigen::ArrayXd y_vals;
    // centralized/scaled 3rd coeff
    Eigen::ArrayXd coef_c;
    // mean/std of x_vals
    double x_mean;
    double x_std;
};

TEST_F(PolyfuncTest, PolyFitting)
{
    // memory map for type conversion
    Eigen::Map<const Eigen::ArrayXd> x_vals_m(x_vals.data(), x_vals.size());
    Eigen::Map<const Eigen::ArrayXd> pn_coef_3rd_m(
            pn_coef_3rd.data(), pn_coef_3rd.size());

    // without any scaling and centering of the input in polyfit
    auto [coef, mean, std] = polyfit(x_vals_m, y_vals, 3);
    EXPECT_NEAR(mean, 0.0, abs_tol) << "MEAN must be zero w/o centralizing!";
    EXPECT_NEAR(std, 1.0, abs_tol)
            << "STD must be unity w/o centralizing and scaling!";
    validate_coeffs(coef, pn_coef_3rd_m,
            std::string("without centralization and scaling"));

    // turn on scaling and centering for the original input in polyfit
    std::tie(coef, mean, std) = polyfit(x_vals_m, y_vals, 3, true);
    EXPECT_NEAR(mean, x_mean, abs_tol) << "Wrong MEAN  w/ centralizing!";
    EXPECT_NEAR(std, x_std, abs_tol)
            << "Wrong STD  w/ centralizing and scaling!";
    validate_coeffs(
            coef, coef_c, std::string("with centralization and scaling"));

    // test exceptions
    // bad deg for polynomial
    EXPECT_THROW(polyfit(x_vals_m, y_vals, 10), isce3::except::InvalidArgument)
            << "Must throw ISCE3 InvalidArgument for degree of polynomial "
               "being too large!";
    // test size mismtach between two vectors
    EXPECT_THROW(
            polyfit(x_vals_m, y_vals.head(4), 3), isce3::except::LengthError)
            << "Must throw ISCE3 LengthError due to size mismtach of two input "
               "vectors!";
}

TEST_F(PolyfuncTest, PolyDerivative)
{
    // memory map for type conversion
    Eigen::Map<const Eigen::ArrayXd> pn_coef_3rd_m(
            pn_coef_3rd.data(), pn_coef_3rd.size());
    Eigen::Map<const Eigen::ArrayXd> pn_coef_2ed_m(
            pn_coef_2ed.data(), pn_coef_2ed.size());

    // w/o scaling (std == 1.0)
    auto coef_der = polyder(pn_coef_3rd_m);
    validate_coeffs(coef_der, pn_coef_2ed_m,
            std::string("for derivative of coeff w/o scaling"));

    // test exceptions
    // bad std value
    EXPECT_THROW(polyder(pn_coef_3rd_m, -0.0), isce3::except::InvalidArgument)
            << "Must throw ISCE3 InvalidArgument for non-positive STD value!";
}

TEST_F(PolyfuncTest, PolyValue)
{
    // memory mapping
    Eigen::Map<const Eigen::ArrayXd> pn_coef_3rd_m(
            pn_coef_3rd.data(), pn_coef_3rd.size());

    // check evaluated values for all input values w/ and w/o centering
    for (std::size_t idx = 0; idx < x_vals.size(); ++idx) {
        // w/o centralization and scaling
        EXPECT_NEAR(polyval(pn_coef_3rd_m, x_vals[idx]), y_vals(idx), abs_tol)
                << "Eval of polynomial (w/o centering) is wrong for input "
                << x_vals[idx];
        // w/ centralization and scaling
        EXPECT_NEAR(polyval(coef_c, x_vals[idx], x_mean, x_std), y_vals(idx),
                abs_tol)
                << "Eval of polynomial (w/ centering) is wrong for input "
                << x_vals[idx];
    }

    // test overloaded polyval with input array
    Eigen::Map<const Eigen::ArrayXd> x_vals_m(x_vals.data(), x_vals.size());
    // w/o centralization and scaling
    auto y = polyval(pn_coef_3rd_m, x_vals_m);
    // w/ centralization and scaling
    auto y_c = polyval(coef_c, x_vals_m, x_mean, x_std);
    for (std::size_t idx = 0; idx < x_vals.size(); ++idx) {
        EXPECT_NEAR(y(idx), y_vals(idx), abs_tol)
                << "Eval of polynomial (w/o centering) for an array is wrong @ "
                   "index "
                << idx << " and for input " << x_vals[idx];
        EXPECT_NEAR(y_c(idx), y_vals(idx), abs_tol)
                << "Eval of polynomial (w/ centering) for an array is wrong @ "
                   "index "
                << idx << " for input " << x_vals[idx];
    }
}

TEST_F(PolyfuncTest, PolyFitObject)
{
    // memory map
    Eigen::Map<const Eigen::ArrayXd> x_vals_m(x_vals.data(), x_vals.size());

    // create Poly1d object  w/ centering and scaling
    auto pf_obj = polyfitObj(x_vals_m, y_vals, 3, true);
    EXPECT_EQ(pf_obj.order, 3) << "Wtrong order of the Poly1d object!";
    EXPECT_NEAR(pf_obj.mean, x_mean, abs_tol)
            << "Wrong MEAN of the Poly1d object!";
    EXPECT_NEAR(pf_obj.norm, x_std, abs_tol)
            << "Wrong STD of the Poly1d object!";
    Eigen::Map<Eigen::ArrayXd> pf_coef(
            pf_obj.coeffs.data(), pf_obj.coeffs.size());
    validate_coeffs(pf_coef, coef_c, std::string("for Poly1d object"));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
