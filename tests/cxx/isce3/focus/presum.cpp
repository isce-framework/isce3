#include <gtest/gtest.h>
#include <isce3/core/Kernels.h>
#include <isce3/focus/Presum.h>


TEST(Presum, Domain)
{
    // Check that coefficients cover the expected domain.

    long n = 10, offset = 0;
    // set up a time vector--uniformly sampled.
    std::vector<double> t(n);
    for (long i = 0; i < n; ++i) {
        t[i] = i;
    }
    // Use a dummy autocorrelation function, with width = 3.
    isce3::core::BartlettKernel<double> acorr(3.0);
    // Compute weights.
    {
        // Expect three coefficients at i = {4, 5, 6}
        auto coeff = isce3::focus::getPresumWeights(acorr, t, 5.0, &offset);
        EXPECT_EQ(coeff.size(), 3);
        EXPECT_EQ(offset, 4);
    }
    {
        // Ask for an output exactly between two points.
        // Expect an extra coeff (four) at i = {4, 5, 6, 7}
        auto coeff = isce3::focus::getPresumWeights(acorr, t, 5.5, &offset);
        EXPECT_EQ(coeff.size(), 4);
        EXPECT_EQ(offset, 4);
    }
}

TEST(Presum, Values)
{
    // Check that coefficients have the expected values.

    long n = 10, offset = 0;
    // set up a time vector--uniformly sampled.
    Eigen::VectorXd t(n);
    for (long i = 0; i < n; ++i) {
        t[i] = i;
    }
    // Use a dummy autocorrelation function, with width = 3.
    isce3::core::BartlettKernel<double> acorr(3.0);
    // Compute weights.
    {
        // Expect three coefficients at i = {4, 5, 6}
        // with values {0, 1, 0}
        auto coeff = isce3::focus::getPresumWeights(acorr, t, 5.0, &offset);
        EXPECT_DOUBLE_EQ(coeff(0), 0.0);
        EXPECT_DOUBLE_EQ(coeff(1), 1.0);
        EXPECT_DOUBLE_EQ(coeff(2), 0.0);
        EXPECT_EQ(coeff.size(), 3);
        EXPECT_EQ(offset, 4);
    }

    // TODO more general example?
}


TEST(Presum, NoData)
{
    // Check behavior for a gap too large.
    // Data correlated on interval [-L,L]
    const double L = 1.0;
    isce3::core::AzimuthKernel<double> acorr(L);
    // Only have two data points, separated by more than that.
    std::vector<double> t {0.0, 10 * L};

    long offset = -1;
    auto coeffs = isce3::focus::getPresumWeights(acorr, t, 5 * L, &offset);
    EXPECT_EQ(coeffs.size(), 0);
    EXPECT_GE(offset, 0);
    EXPECT_LT(offset, t.size());
}


int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
