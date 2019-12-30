#include <gtest/gtest.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <vector>

#include <isce/cuda/except/Error.h>
#include <isce/cuda/fft/FFTPlan.h>

#include "FFTTestHelper.h"

using isce::cuda::fft::FwdFFTPlan;
using isce::cuda::fft::InvFFTPlan;

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
        thrust::device_vector<thrust::complex<double>> in(n), out(n);

        FwdFFTPlan<double> plan(out.data().get(), in.data().get(), n);
        EXPECT_TRUE( plan );
    }
}

TEST(FFTPlanTest, InvalidPlanConfig)
{
    int n = 15;
    thrust::device_vector<thrust::complex<double>> in(n), out(n);

    // invalid FFT size (must be > 0)
    EXPECT_THROW( { FwdFFTPlan<double> plan(out.data().get(), in.data().get(), -n); },
            isce::cuda::except::CudaError<cufftResult> );
}

struct FFTPlanTest : public testing::TestWithParam<int> {};

TEST_P(FFTPlanTest, FFT1D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n), expected(n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n);

    FwdFFTPlan<double> plan(d_out.data().get(), d_in.data().get(), n);
    plan.execute();
    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_c2c_1d(expected.data(), in.data(), n);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, InverseFFT1D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n), expected(n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n);

    InvFFTPlan<double> plan(d_out.data().get(), d_in.data().get(), n);
    plan.execute();
    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    inv_dft_c2c_1d(expected.data(), in.data(), n);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, RealFFT1D)
{
    int n = GetParam();
    std::vector<double> in(n);
    std::vector<thrust::complex<double>> expected(n/2 + 1);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<double> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n/2 + 1);

    FwdFFTPlan<double> plan(d_out.data().get(), d_in.data().get(), n);
    plan.execute();
    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_r2c_1d(expected.data(), in.data(), n);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, HermitianInverseFFT1D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n/2 + 1);
    std::vector<double> expected(n);

    // create input data using DFT so it has Hermitian symmetry
    {
        std::vector<double> x(n);

        RealUniformDistribution<double> U(0., 1.);
        for (int i = 0; i < n; ++i) { x[i] = U.sample(); }

        fwd_dft_r2c_1d(in.data(), x.data(), n);
    }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<double> d_out(n);

    InvFFTPlan<double> plan(d_out.data().get(), d_in.data().get(), n);
    plan.execute();
    std::vector<double> out = copyToHost(d_out);

    inv_dft_c2r_1d(expected.data(), in.data(), n);
    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, FFT2D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n * n), expected(n * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n * n);

    FwdFFTPlan<double> plan(d_out.data().get(), d_in.data().get(), {n, n});
    plan.execute();
    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_c2c_2d(expected.data(), in.data(), {n, n});
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, InverseFFT2D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n * n), expected(n * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n * n);

    InvFFTPlan<double> plan(d_out.data().get(), d_in.data().get(), {n, n});
    plan.execute();
    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    inv_dft_c2c_2d(expected.data(), in.data(), {n, n});
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, RealFFT2D)
{
    int n = GetParam();
    std::vector<double> in(n * n);
    std::vector<thrust::complex<double>> expected(n * n);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<double> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n * n);

    FwdFFTPlan<double> plan(d_out.data().get(), d_in.data().get(), {n, n});
    plan.execute();
    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_r2c_2d(expected.data(), in.data(), {n, n});
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTPlanTest, HermitianInverseFFT2D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n * n);
    std::vector<double> expected(n * n);

    // create input data using DFT so it has Hermitian symmetry
    {
        std::vector<double> x(n * n);

        RealUniformDistribution<double> U(0., 1.);
        for (int i = 0; i < n * n; ++i) { x[i] = U.sample(); }

        fwd_dft_r2c_2d(in.data(), x.data(), {n, n});
    }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<double> d_out(n * n);

    InvFFTPlan<double> plan(d_out.data().get(), d_in.data().get(), {n, n});
    plan.execute();
    std::vector<double> out = copyToHost(d_out);

    inv_dft_c2r_2d(expected.data(), in.data(), {n, n});
    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

// instantiate tests with odd/even FFT sizes
INSTANTIATE_TEST_SUITE_P(FFTPlanTest, FFTPlanTest, testing::Values(15, 16));

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
