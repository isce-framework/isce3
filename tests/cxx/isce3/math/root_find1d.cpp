#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <isce3/core/Common.h>
#include <isce3/core/Poly1d.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/except/Error.h>
#include <isce3/math/RootFind1dBracket.h>
#include <isce3/math/RootFind1dNewton.h>
#include <isce3/math/RootFind1dSecant.h>
#include <isce3/math/detail/RootFind1dBase.h>

using namespace isce3::math;
using namespace isce3::error;

struct RootFind1dTest : public ::testing::Test {

    void SetUp() override
    {
        // Define 1-D poly-fit object
        // (x - 1)^3  -> solution = 1.0
        pf_obj = isce3::core::Poly1d(3, 0.0, 1.0);
        pf_obj.coeffs = std::vector<double> {-1, 3, -3, 1};
        solution = 1;
    }
    // list of methods
    void validate_constrcutor(const detail::RootFind1dBase& rf_obj,
            double f_tol, int iter, std::optional<double> x_tol,
            const std::string& err_msg = {})
    {
        ASSERT_EQ(rf_obj.max_num_iter(), iter)
                << "Wrong max iteration " + err_msg;
        ASSERT_NEAR(rf_obj.func_tol(), f_tol, abs_tol)
                << "Wrong function tolerance " + err_msg;
        if (x_tol) {
            ASSERT_NEAR(*rf_obj.var_tol(), *x_tol, abs_tol)
                    << "Wrong variable tolerance " + err_msg;
        }
    }
    void validate_root(const std::tuple<double, double, bool, int>& rf_result,
            const std::string& err_msg = {})
    {
        auto [x_val, f_val, flag, n_iter] = rf_result;
        EXPECT_LE(n_iter, max_iter)
                << "Exceed max number of iteration " + err_msg;
        EXPECT_TRUE(flag) << "Wrong convergence flag " + err_msg;
        EXPECT_NEAR(x_val, solution, solution * rel_tol)
                << "Wrong root " + err_msg;
        EXPECT_NEAR(f_val, 0.0, f_tol) << "Wrong function value " + err_msg;
    }
    // list of members
    const double f_tol {1e-6};
    const double x_tol {1e-4};
    const int max_iter {30};
    const double rel_tol {1e-3};
    const double abs_tol {1e-8};
    double solution;
    isce3::core::Poly1d pf_obj;
};

TEST_F(RootFind1dTest, NewtonConstructors)
{
    // Validate constructors with correct inputs
    validate_constrcutor(RootFind1dNewton(), 1e-5, 20, {},
            std::string("for default constructor"));

    auto rf_obj = RootFind1dNewton(f_tol, max_iter, x_tol);
    validate_constrcutor(rf_obj, f_tol, max_iter, x_tol,
            std::string("for three-value constructor"));

    rf_obj = RootFind1dNewton(max_iter);
    validate_constrcutor(rf_obj, 1e-5, max_iter, {},
            std::string("for single-value max-iter constructor"));

    // Check exceptions thrown by the constructors with bad input
    EXPECT_THROW(RootFind1dNewton(0), isce3::except::InvalidArgument)
            << "Must throw ISCE3 InvalidArgument for bad max_iter";

    EXPECT_THROW(RootFind1dNewton(0.0), isce3::except::InvalidArgument)
            << "Must throw ISCE3 InvalidArgument for bad func_tol";

    EXPECT_THROW(
            RootFind1dNewton(1e-2, 10, 0.0), isce3::except::InvalidArgument)
            << "Must throw ISCE3 InvalidArgument for bad var_tol";
}

TEST_F(RootFind1dTest, SecantConstructors)
{
    // Validate constructors with correct inputs
    validate_constrcutor(RootFind1dSecant(), 1e-5, 20, {},
            std::string("for default constructor"));

    auto rf_obj = RootFind1dSecant(f_tol, max_iter, x_tol);
    validate_constrcutor(rf_obj, f_tol, max_iter, x_tol,
            std::string("for three-value constructor"));

    rf_obj = RootFind1dSecant(max_iter);
    validate_constrcutor(rf_obj, 1e-5, max_iter, {},
            std::string("for single-value max-iter constructor"));

    // Check exceptions thrown by the constructors with bad input
    EXPECT_THROW(RootFind1dSecant(0), isce3::except::InvalidArgument)
            << "Must throw ISCE3 InvalidArgument for bad max_iter";

    EXPECT_THROW(RootFind1dSecant(0.0), isce3::except::InvalidArgument)
            << "Must throw ISCE3 InvalidArgument for bad func_tol";

    EXPECT_THROW(
            RootFind1dSecant(1e-2, 10, 0.0), isce3::except::InvalidArgument)
            << "Must throw ISCE3 InvalidArgument for bad var_tol";
}

TEST_F(RootFind1dTest, Poly2funcMethod)
{
    auto func = detail::RootFind1dBase::poly2func(pf_obj);
    // Min four distinct values needed to confirm 3rd-order Polynomial
    std::vector<double> x_vals {-2, -1, 0, 1, 2};
    for (const auto& x : x_vals)
        EXPECT_NEAR(func(x), pf_obj.eval(x), abs_tol)
                << "Wrong function value for x=" << x;
}

TEST_F(RootFind1dTest, NewtonRootMethod)
{
    // form RootFind1d obj used in all "root" cases
    auto rf_obj = RootFind1dNewton(f_tol, max_iter, x_tol);

    // build function and its derivative
    auto func = rf_obj.poly2func(pf_obj);
    auto func_der = rf_obj.poly2func(pf_obj.derivative());

    // Root via Newton approach with Poly1d obj
    validate_root(rf_obj.root(pf_obj, 0.0),
            std::string("for Root method with Poly1d as input"));

    // Root via Newton approach with callback function and its derivative
    validate_root(rf_obj.root(func, func_der, 0.0),
            std::string(
                    "for Root method with two callback functions as input"));
}

TEST_F(RootFind1dTest, SecantRootMethod)
{
    // form RootFind1d obj used in all "root" cases
    auto rf_obj = RootFind1dSecant(f_tol, max_iter, x_tol);

    // build function
    auto func = rf_obj.poly2func(pf_obj);

    // Root via Secant approach with callback function (no derivative) and two
    // initial values
    validate_root(rf_obj.root(func, 0.0, -1.0),
            std::string("for Root method with one callback function and two "
                        "initial values"));
}

// easy, right?
CUDA_HOSTDEV double f1(double x)
{
    return std::cos(x);
}

// problem #3 from Alefeld 1995
CUDA_HOSTDEV double f2(double x) {
    // const double a = -40.0, b = -1.0;
    const double a = -100.0, b = -2.0;
    return a * x * std::exp(b * x);
}

// Eq. 4.1 in Brent 1973
// also problem #13 in Alefeld 1995
CUDA_HOSTDEV double f3(double x) {
    if (std::abs(x) < 3.8e-4) {
        return 0.0;
    }
    // See Alefeld 1995 for note on underflow.
    // return x * std::exp(-std::pow(x, -2));
    return x / std::exp(std::pow(x, -2));
}

// problem from Ch. 4 table 4.1 in Brent 1973
CUDA_HOSTDEV double f4(double x) {
    return std::pow(x, 9.0);
}

// test behavior wrt NaN values
CUDA_HOSTDEV double func_with_nan(double x) {
    if (std::abs(x) < 1e-3) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::pow(x, 3.0);
}

TEST(find_zero, brent)
{
    double root = 0.0;
    ErrorCode status;
    double tol = 1e-12;

    status = find_zero_brent(4.0, 5.0, f1, tol, &root);
    EXPECT_EQ(status, ErrorCode::Success);
    EXPECT_NEAR(3 * M_PI / 2, root, tol);

    status = find_zero_brent(-9.0, 31.0, f2, tol, &root);
    EXPECT_EQ(status, ErrorCode::Success);
    EXPECT_NEAR(0.0, root, tol);

    status = find_zero_brent(-1.0, 4.0, f3, tol, &root);
    EXPECT_EQ(status, ErrorCode::Success);
    // This problem still underflows on my machine.
    if (std::abs(f3(root)) > 0.0) {
        EXPECT_NEAR(0.0, root, tol);
    }

    status = find_zero_brent(-1.0, 1.1, f4, tol, &root);
    EXPECT_EQ(status, ErrorCode::Success);
    EXPECT_NEAR(0.0, root, tol);

    status = find_zero_brent(-5.0, 5.0, func_with_nan, tol, &root);
    EXPECT_EQ(status, ErrorCode::FailedToConverge);
}

TEST(find_zero, bisection)
{
    double root = 0.0;
    ErrorCode status;
    double tol = 1e-12;

    status = find_zero_bisection(4.0, 5.0, f1, tol, &root);
    EXPECT_EQ(status, ErrorCode::Success);
    EXPECT_NEAR(3 * M_PI / 2, root, tol);

    status = find_zero_bisection(-9.0, 31.0, f2, tol, &root);
    EXPECT_EQ(status, ErrorCode::Success);
    EXPECT_NEAR(0.0, root, tol);

    status = find_zero_bisection(-1.0, 4.0, f3, tol, &root);
    EXPECT_EQ(status, ErrorCode::Success);
    // This problem still underflows on my machine.
    if (std::abs(f3(root)) > 0.0) {
        printf("f3(x) = %g\n", f3(root));
        EXPECT_NEAR(0.0, root, tol);
    }

    status = find_zero_bisection(-1.0, 1.1, f4, tol, &root);
    EXPECT_EQ(status, ErrorCode::Success);
    EXPECT_NEAR(0.0, root, tol);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
