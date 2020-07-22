#include <complex>
#include <gtest/gtest.h>
#include <vector>

#include <isce3/except/Error.h>
#include <isce3/fft/FFT.h>

#include "FFTTestHelper.h"

TEST(FFTTest, InvalidAxis)
{
    int n = 15;
    int batch = 8;
    std::vector<std::complex<double>> in(batch * n), out(batch * n);

    // axis out-of-range for 2-D data
    EXPECT_THROW( { isce3::fft::fft1d(out.data(), in.data(), {n, batch}, 3); }, isce3::except::OutOfRange );
}

struct PlanFFTTest : public testing::TestWithParam<int> {};

TEST_P(PlanFFTTest, FFT1D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n), out(n), expected(n);

    auto plan = isce3::fft::planfft1d(out.data(), in.data(), n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_1d(expected.data(), in.data(), n);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, RealFFT1D)
{
    int n = GetParam();
    std::vector<double> in(n);
    std::vector<std::complex<double>> out(n/2 + 1), expected(n/2 + 1);

    auto plan = isce3::fft::planfft1d(out.data(), in.data(), n);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    fwd_dft_r2c_1d(expected.data(), in.data(), n);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, FFTAxis0)
{
    int n = GetParam();
    int batch = 8;
    std::vector<std::complex<double>> in(n * batch), out(n * batch), expected(n * batch);

    auto plan = isce3::fft::planfft1d(out.data(), in.data(), {n, batch}, 0);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * batch; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_1d(expected.data(), in.data(), n, batch, batch, 1);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, FFTAxis1)
{
    int n = GetParam();
    int batch = 8;
    std::vector<std::complex<double>> in(batch * n), out(batch * n), expected(batch * n);

    auto plan = isce3::fft::planfft1d(out.data(), in.data(), {batch, n}, 1);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < batch * n; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_1d(expected.data(), in.data(), n, 1, batch, n);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, FFT2D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n * n), out(n * n), expected(n * n);

    auto plan = isce3::fft::planfft2d(out.data(), in.data(), {n, n});

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_2d(expected.data(), in.data(), {n, n});
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, RealFFT2D)
{
    int n = GetParam();
    std::vector<double> in(n * n);
    std::vector<std::complex<double>> out(n * n), expected(n * n);

    auto plan = isce3::fft::planfft2d(out.data(), in.data(), {n, n});

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    fwd_dft_r2c_2d(expected.data(), in.data(), {n, n});
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, InverseFFT1D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n), out(n), expected(n);

    auto plan = isce3::fft::planifft1d(out.data(), in.data(), n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_1d(expected.data(), in.data(), n);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, HermitianInverseFFT1D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n/2 + 1);
    std::vector<double> out(n), expected(n);

    auto plan = isce3::fft::planifft1d(out.data(), in.data(), n);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n/2 + 1; ++i) { in[i] = U.sample(); }

    inv_dft_c2r_1d(expected.data(), in.data(), n);
    plan.execute();
    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, InverseFFTAxis0)
{
    int n = GetParam();
    int batch = 8;
    std::vector<std::complex<double>> in(n * batch), out(n * batch), expected(n * batch);

    auto plan = isce3::fft::planifft1d(out.data(), in.data(), {n, batch}, 0);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * batch; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_1d(expected.data(), in.data(), n, batch, batch, 1);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, InverseFFTAxis1)
{
    int n = GetParam();
    int batch = 8;
    std::vector<std::complex<double>> in(batch * n), out(batch * n), expected(batch * n);

    auto plan = isce3::fft::planifft1d(out.data(), in.data(), {batch, n}, 1);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < batch * n; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_1d(expected.data(), in.data(), n, 1, batch, n);
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, InverseFFT2D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n * n), out(n * n), expected(n * n);

    auto plan = isce3::fft::planifft2d(out.data(), in.data(), {n, n});

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_2d(expected.data(), in.data(), {n, n});
    plan.execute();
    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(PlanFFTTest, HermitianInverseFFT2D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n * n);
    std::vector<double> out(n * n), expected(n * n);

    auto plan = isce3::fft::planifft2d(out.data(), in.data(), {n, n});

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    inv_dft_c2r_2d(expected.data(), in.data(), {n, n});
    plan.execute();
    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

// instantiate tests with odd/even FFT sizes
INSTANTIATE_TEST_SUITE_P(PlanFFTTest, PlanFFTTest, testing::Values(15, 16));

struct FFTTest : public testing::TestWithParam<int> {};

TEST_P(FFTTest, FFT1D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n), out(n), expected(n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_1d(expected.data(), in.data(), n);

    isce3::fft::fft1d(out.data(), in.data(), n);

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, RealFFT1D)
{
    int n = GetParam();
    std::vector<double> in(n);
    std::vector<std::complex<double>> out(n/2 + 1), expected(n/2 + 1);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    fwd_dft_r2c_1d(expected.data(), in.data(), n);

    isce3::fft::fft1d(out.data(), in.data(), n);

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, FFTAxis0)
{
    int n = GetParam();
    int batch = 8;
    std::vector<std::complex<double>> in(n * batch), out(n * batch), expected(n * batch);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * batch; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_1d(expected.data(), in.data(), n, batch, batch, 1);

    isce3::fft::fft1d(out.data(), in.data(), {n, batch}, 0);

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, FFTAxis1)
{
    int n = GetParam();
    int batch = 8;
    std::vector<std::complex<double>> in(batch * n), out(batch * n), expected(batch * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < batch * n; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_1d(expected.data(), in.data(), n, 1, batch, n);

    isce3::fft::fft1d(out.data(), in.data(), {batch, n}, 1);

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, FFT2D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n * n), out(n * n), expected(n * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    fwd_dft_c2c_2d(expected.data(), in.data(), {n, n});

    isce3::fft::fft2d(out.data(), in.data(), {n, n});

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, RealFFT2D)
{
    int n = GetParam();
    std::vector<double> in(n * n);
    std::vector<std::complex<double>> out(n * n), expected(n * n);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    fwd_dft_r2c_2d(expected.data(), in.data(), {n, n});

    isce3::fft::fft2d(out.data(), in.data(), {n, n});

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, InverseFFT1D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n), out(n), expected(n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_1d(expected.data(), in.data(), n);

    isce3::fft::ifft1d(out.data(), in.data(), n);

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, HermitianInverseFFT1D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n/2 + 1);
    std::vector<double> out(n), expected(n);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n/2 + 1; ++i) { in[i] = U.sample(); }

    inv_dft_c2r_1d(expected.data(), in.data(), n);

    isce3::fft::ifft1d(out.data(), in.data(), n);

    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

TEST_P(FFTTest, InverseFFTAxis0)
{
    int n = GetParam();
    int batch = 8;
    std::vector<std::complex<double>> in(n * batch), out(n * batch), expected(n * batch);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * batch; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_1d(expected.data(), in.data(), n, batch, batch, 1);

    isce3::fft::ifft1d(out.data(), in.data(), {n, batch}, 0);

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, InverseFFTAxis1)
{
    int n = GetParam();
    int batch = 8;
    std::vector<std::complex<double>> in(batch * n), out(batch * n), expected(batch * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < batch * n; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_1d(expected.data(), in.data(), n, 1, batch, n);

    isce3::fft::ifft1d(out.data(), in.data(), {batch, n}, 1);

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, InverseFFT2D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n * n), out(n * n), expected(n * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    inv_dft_c2c_2d(expected.data(), in.data(), {n, n});

    isce3::fft::ifft2d(out.data(), in.data(), {n, n});

    EXPECT_PRED3( compareVectors<std::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, HermitianInverseFFT2D)
{
    int n = GetParam();
    std::vector<std::complex<double>> in(n * n);
    std::vector<double> out(n * n), expected(n * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    inv_dft_c2r_2d(expected.data(), in.data(), {n, n});

    isce3::fft::ifft2d(out.data(), in.data(), {n, n});

    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

// instantiate tests with odd/even FFT sizes
INSTANTIATE_TEST_SUITE_P(FFTTest, FFTTest, testing::Values(15, 16));

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
