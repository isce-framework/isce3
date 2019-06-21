/*
 * These tests compare NFFT and its adjoint to the equivalent DFT.
 * See the interp1d test for comparison of NFFT to other interpolation methods.
 */
#include <complex>
#include <random>
#include <gtest/gtest.h>
#include "isce/signal/NFFT.h"
#include "isce/signal/Filter.h"

const int seed = 1234;
using namespace std::literals::complex_literals;
using isce::signal::fftfreq;
using std::sqrt;

#ifndef _DBG_NFFT
#define _DBG_NFFT 0
#endif

// Bound from NFFT ACM TMS paper for Kaiser-Bessel window function.
double
error_bound(long m, long n, long fft_size)
{
    double x = sqrt(1.0 - n * 1.0 / fft_size);
    return 4*M_PI*(sqrt(m)+m) * sqrt(x) * std::exp(-2*M_PI*m*x);
}

void
compare_output(const isce::signal::NFFT<double> &p,
               const std::valarray<std::complex<double>> &expected,
               const std::valarray<std::complex<double>> &result)
{
    size_t n = expected.size();
    double rms=0, dxmax=0;
    for (size_t i=0; i<n; i++) {
        auto dx = std::abs(result[i] - expected[i]);
        dxmax = (dx > dxmax) ? dx : dxmax;
        rms += dx * dx;
    }
    rms = sqrt(rms / n);
    long m = (p.size_kernel() - 1) / 2;
    double bound = error_bound(m, p.size_spectrum(), p.size_transform());
    if (_DBG_NFFT) {
        printf("RMS error = %.16g\n", rms);
        printf("max error = %.16g\n", dxmax);
        printf("error bound factor = %.16g\n", bound);
    }
    // Assume L1 norm of spectrum is roughly == N.
    EXPECT_LE(dxmax, p.size_spectrum()*bound);
}

void
test_nfft(size_t m, size_t nf, size_t fft_size)
{
    // Generate test data.
    const size_t nt = 256;
    std::valarray<std::complex<double>> xt_ref(nt), xt(nt), xf(nf);
    std::valarray<double> times(nt);
    
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform(0.0, nf-1.0);

    for (size_t i=0; i<nt; ++i) {
        times[i] = uniform(rng);
    }
    for (size_t i=0; i<nf; ++i) {
        xf[i] = normal(rng) + 1i * normal(rng);
    }
    
    std::valarray<double> f(nf);
    fftfreq(1.0, f);

    // Compute DFT.
    for (long i=0; i<nt; ++i) {
        xt_ref[i] = 0.0;
        for (long k=0; k<nf; ++k) {
            double phase = 2.0 * M_PI * f[k] * times[i];
            auto w = std::cos(phase) + 1i * std::sin(phase);
            xt_ref[i] += w * xf[k];
        }
        xt_ref[i] *= 1.0/nf;
    }

    // Compute NFFT.
    isce::signal::NFFT<double> nfft(m, nf, fft_size);
    nfft.execute(xf, times, xt);

    // Compare NFFT to DFT.
    compare_output(nfft, xt_ref, xt);
}

void
test_adjoint_nfft(size_t m, size_t nf, size_t fft_size)
{
    // Generate test data.
    std::valarray<std::complex<double>> xt(nf), xf(nf), xf_ref(nf);
    std::valarray<double> times(nf);
    
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (size_t i=0; i<nf; ++i) {
        // Make sure we cover the time domain, just add some jitter.
        times[i] = (double)i + normal(rng) * 0.05;
        // White noise signal.
        xt[i] = normal(rng) + 1i * normal(rng);
        if (_DBG_NFFT) printf("data %25.16e %25.16e %25.16e\n", times[i],
                              std::real(xt[i]), std::imag(xt[i]));
    }
    
    std::valarray<double> f(nf);
    fftfreq(1.0, f);

    // Convolve data with each tone.
    for (long k=0; k<nf; ++k) {
        xf_ref[k] = 0.0;
        for (long i=0; i<nf; ++i) {
            double phase = -2.0 * M_PI * f[k] * times[i];
            auto w = std::cos(phase) + 1i * std::sin(phase);
            xf_ref[k] += w * xt[i];
        }
    }

    // Compute adjoint NFFT.
    isce::signal::NFFT<double> nfft(m, nf, fft_size);
    nfft.execute_adjoint(xt, times, xf);

    // Compare adjoint NFFT to expected values.
    // Not really sure if the same error bound applies to adjoint transform.
    compare_output(nfft, xf_ref, xf);
}

TEST(NFFT, ShortEven) { test_nfft(1,   8,   32); }
TEST(NFFT, LongEven)  { test_nfft(4, 256, 1024); }
TEST(NFFT, LongOdd)   { test_nfft(4, 256,  625); }
TEST(NFFT, Odd)
{
    ASSERT_THROW(isce::signal::NFFT<double>(1, 9, 32),
                 isce::except::LengthError);
}

TEST(AdjointNFFT, ShortEven) { test_adjoint_nfft(1,   8,   32); }
TEST(AdjointNFFT, MedEven)   { test_adjoint_nfft(2, 256, 1024); }
TEST(AdjointNFFT, LongEven)  { test_adjoint_nfft(4, 256, 1024); }
TEST(AdjointNFFT, LongOdd)   { test_adjoint_nfft(4, 256,  625); }

TEST(Kernel, Singularity)
{
    size_t m = 1;
    auto window = isce::core::NFFTKernel<double>(m, 10, 20);
    double dx = 1e-6;
    // Window should be monotonically decreasing.
    EXPECT_GT(window(m-dx), window(m));
    EXPECT_GT(window(m), window(m+dx));
}

int
main(int argc, char *argv[])
{
      testing::InitGoogleTest(&argc, argv);
      return RUN_ALL_TESTS();
}
