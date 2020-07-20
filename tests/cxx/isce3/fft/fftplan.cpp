#include <complex>
#include <gtest/gtest.h>
#include <vector>

#include <isce3/except/Error.h>
#include <isce3/fft/FFTPlan.h>

#include "FFTTestHelper.h"

using isce::fft::FwdFFTPlan;
using isce::fft::InvFFTPlan;

TEST(FFTPlanTest, Logical)
{
    // invalid (uninitialized) plan
    {
        FwdFFTPlan<double> plan;
        EXPECT_FALSE( plan );
    }

    // valid (initialized) plan
    {
        int n = 15;
        std::vector<std::complex<double>> in(n), out(n);

        FwdFFTPlan<double> plan(out.data(), in.data(), n);
        EXPECT_TRUE( plan );
    }
}

TEST(FFTPlanTest, InvalidPlanConfig)
{
    int n = 15;
    std::vector<std::complex<double>> in(n), out(n);

    // invalid FFT size (must be > 0)
    EXPECT_THROW( { FwdFFTPlan<double> plan(out.data(), in.data(), -n); }, isce::except::RuntimeError );
}

struct FFTPlanTest : public testing::TestWithParam<int> {};

TEST_P(FFTPlanTest, FFT1D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n), out(n), expected(n);

    FwdFFTPlan<double> plan(out.data(), in.data(), n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_1d(expected.data(), in.data(), n);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, InverseFFT1D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n), out(n), expected(n);

    InvFFTPlan<double> plan(out.data(), in.data(), n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_1d(expected.data(), in.data(), n);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, RealFFT1D)
{
    int n = GetParam();
    std::vector<double> in(n);
    std::vector<std::complex<double>> out(n/2 + 1), expected(n/2 + 1);

    FwdFFTPlan<double> plan(out.data(), in.data(), n);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    fwd_dft_r2c_1d(expected.data(), in.data(), n);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, HermitianInverseFFT1D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n/2 + 1);
    std::vector<double> out(n), expected(n);

    InvFFTPlan<double> plan(out.data(), in.data(), n);

    // create input data using DFT so it has Hermitian symmetry
    {
        std::vector<double> x(n);

        RealUniformDistribution<double> U(0., 1.);
        for (int i = 0; i < n; ++i) { x[i] = U.sample(); }

        fwd_dft_r2c_1d(in.data(), x.data(), n);
    }

    inv_dft_c2r_1d(expected.data(), in.data(), n);
    plan.execute();
    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, FFT2D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n * n), out(n * n), expected(n * n);

    FwdFFTPlan<double> plan(out.data(), in.data(), {n, n});

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_2d(expected.data(), in.data(), {n, n});
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, InverseFFT2D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n * n), out(n * n), expected(n * n);

    InvFFTPlan<double> plan(out.data(), in.data(), {n, n});

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_2d(expected.data(), in.data(), {n, n});
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, RealFFT2D)
{
    int n = GetParam();
    std::vector<double> in(n * n);
    std::vector<std::complex<double>> out(n * n), expected(n * n);

    FwdFFTPlan<double> plan(out.data(), in.data(), {n, n});

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    fwd_dft_r2c_2d(expected.data(), in.data(), {n, n});
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, HermitianInverseFFT2D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n * n);
    std::vector<double> out(n * n), expected(n * n);

    InvFFTPlan<double> plan(out.data(), in.data(), {n, n});

    // create input data using DFT so it has Hermitian symmetry
    {
        std::vector<double> x(n * n);

        RealUniformDistribution<double> U(0., 1.);
        for (int i = 0; i < n * n; ++i) { x[i] = U.sample(); }

        fwd_dft_r2c_2d(in.data(), x.data(), {n, n});
    }

    inv_dft_c2r_2d(expected.data(), in.data(), {n, n});
    plan.execute();
    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

// instantiate tests with odd/even FFT sizes
INSTANTIATE_TEST_SUITE_P(FFTPlanTest, FFTPlanTest, testing::Values(15, 16));

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
