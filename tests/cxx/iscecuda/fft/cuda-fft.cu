#include <gtest/gtest.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <vector>

#include <isce/except/Error.h>
#include <isce/cuda/fft/FFT.h>

#include "FFTTestHelper.h"

TEST(FFTTest, InvalidAxis)
{
    int n = 15;
    int batch = 8;
    thrust::device_vector<thrust::complex<double>> in(batch * n), out(batch * n);

    // axis out-of-range for 2-D data
    EXPECT_THROW( { isce::cuda::fft::fft1d(out.data().get(), in.data().get(), {n, batch}, 3); },
            isce::except::OutOfRange );
}

struct FFTTest : public testing::TestWithParam<int> {};

TEST_P(FFTTest, FFT1D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n), expected(n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n);

    isce::cuda::fft::fft1d(d_out.data().get(), d_in.data().get(), n);

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_c2c_1d(expected.data(), in.data(), n);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, RealFFT1D)
{
    int n = GetParam();
    std::vector<double> in(n);
    std::vector<thrust::complex<double>> expected(n/2 + 1);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<double> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n/2 + 1);

    isce::cuda::fft::fft1d(d_out.data().get(), d_in.data().get(), n);

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_r2c_1d(expected.data(), in.data(), n);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, FFTAxis0)
{
    int n = GetParam();
    int batch = 8;
    std::vector<thrust::complex<double>> in(n * batch), expected(n * batch);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * batch; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n * batch);

    isce::cuda::fft::fft1d(d_out.data().get(), d_in.data().get(), {n, batch}, 0);

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_c2c_1d(expected.data(), in.data(), n, batch, batch, 1);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, FFTAxis1)
{
    int n = GetParam();
    int batch = 8;
    std::vector<thrust::complex<double>> in(batch * n), expected(batch * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < batch * n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(batch * n);

    isce::cuda::fft::fft1d(d_out.data().get(), d_in.data().get(), {batch, n}, 1);

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_c2c_1d(expected.data(), in.data(), n, 1, batch, n);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, FFT2D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n * n), expected(n * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n * n);

    isce::cuda::fft::fft2d(d_out.data().get(), d_in.data().get(), {n, n});

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_c2c_2d(expected.data(), in.data(), {n, n});
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, RealFFT2D)
{
    int n = GetParam();
    std::vector<double> in(n * n);
    std::vector<thrust::complex<double>> expected(n * n);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<double> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n * n);

    isce::cuda::fft::fft2d(d_out.data().get(), d_in.data().get(), {n, n});

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    fwd_dft_r2c_2d(expected.data(), in.data(), {n, n});
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, InverseFFT1D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n), expected(n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n);

    isce::cuda::fft::ifft1d(d_out.data().get(), d_in.data().get(), n);

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    inv_dft_c2c_1d(expected.data(), in.data(), n);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, HermitianInverseFFT1D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n/2 + 1);
    std::vector<double> expected(n);

    RealUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n/2 + 1; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<double> d_out(n);

    isce::cuda::fft::ifft1d(d_out.data().get(), d_in.data().get(), n);

    std::vector<double> out = copyToHost(d_out);

    inv_dft_c2r_1d(expected.data(), in.data(), n);
    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

TEST_P(FFTTest, InverseFFTAxis0)
{
    int n = GetParam();
    int batch = 8;
    std::vector<thrust::complex<double>> in(n * batch), expected(n * batch);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * batch; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n * batch);

    isce::cuda::fft::ifft1d(d_out.data().get(), d_in.data().get(), {n, batch}, 0);

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    inv_dft_c2c_1d(expected.data(), in.data(), n, batch, batch, 1);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, InverseFFTAxis1)
{
    int n = GetParam();
    int batch = 8;
    std::vector<thrust::complex<double>> in(batch * n), expected(batch * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < batch * n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(batch * n);

    isce::cuda::fft::ifft1d(d_out.data().get(), d_in.data().get(), {batch, n}, 1);

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    inv_dft_c2c_1d(expected.data(), in.data(), n, 1, batch, n);
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, InverseFFT2D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n * n), expected(n * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<thrust::complex<double>> d_out(n * n);

    isce::cuda::fft::ifft2d(d_out.data().get(), d_in.data().get(), {n, n});

    std::vector<thrust::complex<double>> out = copyToHost(d_out);

    inv_dft_c2c_2d(expected.data(), in.data(), {n, n});
    EXPECT_PRED3( compareVectors<thrust::complex<double>>, out, expected, 1e-8 );
}

TEST_P(FFTTest, HermitianInverseFFT2D)
{
    int n = GetParam();
    std::vector<thrust::complex<double>> in(n * n);
    std::vector<double> expected(n * n);

    ComplexUniformDistribution<double> U(0., 1.);
    for (int i = 0; i < n * n; ++i) { in[i] = U.sample(); }

    thrust::device_vector<thrust::complex<double>> d_in = copyToDevice(in);
    thrust::device_vector<double> d_out(n * n);

    isce::cuda::fft::ifft2d(d_out.data().get(), d_in.data().get(), {n, n});

    std::vector<double> out = copyToHost(d_out);

    inv_dft_c2r_2d(expected.data(), in.data(), {n, n});
    EXPECT_PRED3( compareVectors<double>, out, expected, 1e-8 );
}

// instantiate tests with odd/even FFT sizes
INSTANTIATE_TEST_SUITE_P(FFTTest, FFTTest, testing::Values(15, 16));

int main(int argc, char * argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
