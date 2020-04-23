//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Brian Hawkins
// Copyright 2019
//

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <complex>
#include <vector>
#include <random>
#include <gtest/gtest.h>

// isce::core
#include <isce/core/Interp1d.h>
#include <isce/core/Kernels.h>
#include <isce/core/Utilities.h>
#include <isce/math/Sinc.h>
#include <isce/signal/NFFT.h>

using isce::core::interp1d;
using isce::math::sinc;

double
rad2deg(double x)
{
    return x * 180.0 / M_PI;
}

double
_correlation(std::vector<std::complex<double>> &x,
             std::vector<std::complex<double>> &y)
{
    double xy=0.0, xx=0.0, yy=0.0;
    int n = std::min(x.size(), y.size());
    for (int i=0; i<n; ++i) {
        auto xi = x[i];
        auto yi = y[i];
        xy += std::abs(xi * std::conj(yi));
        xx += std::real(xi * std::conj(xi));
        yy += std::real(yi * std::conj(yi));
    }
    return xy / std::sqrt(xx * yy);
}

double
_mean(std::vector<double> &v)
{
    double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
    return sum / v.size();
}

double
_stdev(std::vector<double> &v)
{
    double m = _mean(v);
    double accum = 0.0;
    std::for_each (std::begin(v), std::end(v), [&](const double d) {
        accum += (d - m) * (d - m);
    });
    return std::sqrt(accum / (v.size() - 1.0));
}

double
_phase_stdev(std::vector<std::complex<double>> &x,
             std::vector<std::complex<double>> &y)
{
    auto n = std::min(x.size(), y.size());
    std::vector<double> phase(n);
    // assume we don't have to worry about wrapping
    for (int i=0; i<n; ++i) {
        phase[i] = std::arg(x[i] * std::conj(y[i]));
    }
    return _stdev(phase);
}

double
dB(double x)
{
    return 10.0 * std::log10(std::abs(x));
}

void
power_bias_stdev(double &bias, double &stdev, double minval,
                 std::vector<std::complex<double>> &a,
                 std::vector<std::complex<double>> &b)
{
    std::vector<double> ratio(0);
    auto n = std::min(a.size(), b.size());
    for (int i=0; i<n; ++i) {
        auto aa = std::abs(a[i]);
        auto ab = std::abs(b[i]);
        if ((aa >= minval) && (ab >= minval)) {
            ratio.push_back(aa / ab);
        }
    }
    n = ratio.size();
    std::vector<double> dbr(n);
    for (int i=0; i<n; ++i) {
        dbr[i] = 2 * dB(ratio[i]);
    }
    bias = 2 * dB(_mean(ratio));
    stdev = _stdev(dbr);
}

/* Adapted from julia code at
 * https://github.jpl.nasa.gov/bhawkins/FIRInterp.jl/blob/master/test/common.jl
 */
class TestSignal {
public:
    TestSignal(int n, double bw, int seed);
    std::complex<double> eval(double t);
private:
    int _n;
    double _bw;
    std::vector<double> _t;
    std::vector<std::complex<double>> _w;
};

TestSignal::TestSignal(int n, double bw, int seed)
{
    assert((0.0 <= bw) && (bw < 1.0));

    _n = n;
    _bw = bw;

    int nt = 4 * _n;
    _t.resize(nt);
    _w.resize(nt);

    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0 * n / nt);
    std::uniform_real_distribution<double> uniform(0.0, n-1.0);

    for (int i=0; i<nt; ++i) {
        // targets at uniform locations in [0,n-1]
        _t[i] = uniform(rng);
        // circular normal distributed weights.
        std::complex<double> z(normal(rng), normal(rng));
        _w[i] = z;
    }
}

std::complex<double>
TestSignal::eval(double t)
{
    std::complex<double> sum(0.0, 0.0);
    auto n = _w.size();
    for (int i=0; i<n; ++i) {
        sum += _w[i] * sinc<double>(_bw * (t - _t[i]));
    }
    return sum;
}

struct Interp1dTest : public ::testing::Test
{
    // Length of input signal.
    const int n = 512;
    // Randon number generator seed so that results are repeatable.
    const int seed = 1234;
    // signal bandwidth as a fraction of sample rate.
    const double bw = 0.8;
    // Trick to get apples-to-apples comparisons even for different kernel
    // widths.  See assertions for acceptable range of values.
    const int pad = 8;
    // Amplitude mask for backscatter.
    const double minval = 1e-6;

    // Generator of bandlimited test signal at arbitrary time samples.
    TestSignal ts;
    // Realization of signal at integer time steps.
    std::valarray<std::complex<double>> signal;
    // Realizations of signal at arbitrary time steps;
    std::vector<std::complex<double>> ref, out;

    protected:
        // Constructor
        Interp1dTest() : ts(n, bw, seed) {
            assert(2*pad < n);  // and pad>0 since it's unsigned.
            // Generate signal at integer sample times.
            signal.resize(n);
            for (int i=0; i<n; ++i) {
                double t = (double)i;
                signal[i] = ts.eval(t);
            }
        }

        void
        fill_ref(const std::vector<double> &times)
        {
            auto nt = times.size();
            ref.assign(nt, 0.0);
            for (int i=0; i<nt; ++i) {
               ref[i] = ts.eval(times[i]);
            }
        }

        void
        fill_out(const std::vector<double> &times,
                 isce::core::Kernel<double> &kernel)
        {
            auto nt = times.size();
            out.assign(nt, 0.0);
            for (int i=0; i<nt; ++i) {
                out[i] = interp1d<double,std::complex<double>>(kernel, signal,
                                                               times[i]);
            }
        }

        void
        mask_edges(const std::vector<double> &times) {
            auto nt = times.size();
            for (int i=0; i<nt; ++i) {
                auto t = times[i];
                if ((t < pad) || (t > n-1-pad)) {
                    out[i] = ref[i] = 0.0;
                }
            }
        }

        // Must have filled out and ref arrays.
        void
        check(double min_cor, double max_phs, double max_bias,
              double max_spread)
        {
            auto cor = _correlation(ref, out);
            auto dphase = _phase_stdev(ref, out);
            double bias, spread;
            power_bias_stdev(bias, spread, minval, ref, out);

            EXPECT_GE(cor, min_cor);
            EXPECT_LE(rad2deg(dphase), max_phs);
            EXPECT_LE(bias, max_bias);
            EXPECT_LE(spread, max_spread);
            printf("min_cor    %9.6f cor    %9.6f\n", min_cor, cor);
            printf("max_phs    %9.6f phs    %9.6f\n", max_phs, rad2deg(dphase));
            printf("max_bias   %9.6f bias   %9.6f\n", max_bias, bias);
            printf("max_spread %9.6f spread %9.6f\n", max_spread, spread);
            printf("\n");
        }

        void
        check_interp(double min_cor, double max_phs, double max_bias,
                     double max_spread, std::vector<double> &times,
                     isce::core::Kernel<double> &kernel)
        {
            fill_ref(times);
            fill_out(times, kernel);
            mask_edges(times);
            check(min_cor, max_phs, max_bias, max_spread);
        }

        std::vector<double>
        gen_rand_times() {
            std::mt19937 rng(2 * seed);
            std::uniform_real_distribution<double> uniform(-0.5, 0.5);
            std::vector<double> times(n);
            for (int i=0; i<n; ++i) {
                times[i] = i + uniform(rng);
            }
            return times;
        }

        void
        test_rand_offsets(double min_cor, double max_phs, double max_bias,
                          double max_spread,
                          isce::core::Kernel<double> &kernel)
        {
            printf("Testing random offsets.\n");
            auto times = gen_rand_times();
            check_interp(min_cor, max_phs, max_bias, max_spread, times, kernel);
        }

        void
        test_fixed_offset(double min_cor, double max_phs, double max_bias,
                         double max_spread,
                         isce::core::Kernel<double> &kernel,
                         double off)
        {
            printf("Testing fixed offset=%g\n", off);
            std::vector<double> times(n);
            for (int i=0; i<n; ++i) {
                times[i] = i + off;
            }
            check_interp(min_cor, max_phs, max_bias, max_spread, times, kernel);
        }
};

TEST_F(Interp1dTest, Linear) {
    auto kernel = isce::core::LinearKernel<double>();
    // offset=0 should give back original data for this kernel.
    test_fixed_offset(0.999999, 0.001, 0.001, 0.001, kernel,  0.0);
    test_fixed_offset(0.95, 30.0, 5.0, 5.0, kernel, -0.3);
    test_fixed_offset(0.95, 30.0, 5.0, 5.0, kernel,  0.3);
    test_fixed_offset(0.95, 40.0, 5.0, 5.0, kernel, -0.5);
    test_fixed_offset(0.95, 40.0, 5.0, 5.0, kernel,  0.5);
    test_rand_offsets(0.95, 30.0, 3.0, 3.0, kernel);
}

TEST_F(Interp1dTest, Knab) {
    auto kernel = isce::core::KnabKernel<double>(9.0, 0.8);
    // offset=0 should give back original data for this kernel.
    test_fixed_offset(0.999999, 0.001, 0.001, 0.001, kernel,  0.0);
    test_fixed_offset(0.998, 5.0, 1.0, 1.0, kernel, -0.3);
    test_fixed_offset(0.998, 5.0, 1.0, 1.0, kernel,  0.3);
    test_fixed_offset(0.998, 5.0, 1.0, 1.0, kernel, -0.5);
    test_fixed_offset(0.998, 5.0, 1.0, 1.0, kernel,  0.5);
    test_rand_offsets(0.998, 5.0, 0.5, 0.5, kernel);
}

TEST_F(Interp1dTest, TabulatedKnab) {
    auto knab = isce::core::KnabKernel<double>(9.0, 0.8);
    auto kernel = isce::core::TabulatedKernel<double>(knab, 2048);
    // offset=0 should give back original data for this kernel.
    test_fixed_offset(0.999999, 0.001, 0.001, 0.001, kernel,  0.0);
    test_rand_offsets(0.998, 5.0, 0.5, 0.5, kernel);
}

TEST_F(Interp1dTest, ChebyKnab) {
    auto knab = isce::core::KnabKernel<double>(9.0, 0.8);
    auto kernel = isce::core::ChebyKernel<double>(knab, 16);
    test_fixed_offset(0.999999, 0.001, 0.001, 0.001, kernel,  0.0);
    test_rand_offsets(0.998, 5.0, 0.5, 0.5, kernel);
}

TEST_F(Interp1dTest, NFFT) {
    // FFT the signal set up by the test class to get a spectrum.
    std::valarray<std::complex<double>> spec(n);
    int i_n = n;
    isce::signal::Signal<double> fft;
    fft.fftPlanForward(signal, spec, 1, &i_n, 1, NULL, 1, 1, NULL, 1, 1, -1);
    fft.forward(signal, spec);

    // Set up NFFT object with 9 taps and 2x oversampling.
    isce::signal::NFFT<double> itp(4, n, 2*n);

    // Feed a spectrum to NFFT object.
    itp.set_spectrum(spec);

    // Use NFFT to interpolate at random times.
    printf("Testing random offsets.\n");
    auto times = gen_rand_times();
    out.assign(times.size(), 0.0);
    for (int i=0; i<times.size(); ++i) {
        out[i] = itp.interp(times[i]);
    }
    // Compare to reference signal.
    fill_ref(times);
    mask_edges(times);
    check(0.998, 1.0, 0.1, 0.1);
}

template<class T>
class SpeedCheck {
public:
    const int n;
    const int npts;
    std::valarray<T> t;
    std::valarray<std::complex<T>> x, y;

    SpeedCheck() : n(1024*512), npts(1024), t(npts), x(npts), y(npts) {
        std::mt19937 rng(2345);
        std::normal_distribution<T> normal(0.0, 1.0);
        std::uniform_real_distribution<T> uniform(0.0, npts-1.0);
        for (int i=0; i<npts; ++i) {
            t[i] = uniform(rng);
            x[i] = std::complex<T>(normal(rng), normal(rng));
        }
    }

    double run(isce::core::Kernel<T> &kernel) {
        std::clock_t start = std::clock();
        for (int i=0; i<n; ++i) {
            int j = i % npts;
            y[j] = isce::core::interp1d(kernel, x, t[j], true);
        }
        return (std::clock() - start) / (double) CLOCKS_PER_SEC;
    }
};

TEST(Kernel, Speed)
{
    using U = float;
    isce::core::NFFTKernel<U> exact(4, 1, 2);
    // A degree 16 Chebyshev and 2k linear table give similar approximation
    // errors of about 5e-8.  See notebook ApproxWindow.ipynb.
    isce::core::ChebyKernel<U> cheby(exact, 16);
    isce::core::TabulatedKernel<U> table(exact, 2048);
    SpeedCheck<U> timer;

    // The first run seems to have a penalty as things get loaded in to cache?
    // So discard its timing result.
    timer.run(table);

    auto time_exact = timer.run(exact);
    printf("exact ran in %g seconds\n", time_exact);
    auto time_cheby = timer.run(cheby);
    printf("cheby ran in %g seconds\n", time_cheby);
    auto time_table = timer.run(table);
    printf("table ran in %g seconds\n", time_table);

    // XXX Concerned that performance isn't portable, and our unit tests should
    // XXX only verify correctness.  Disabled until we have a separate
    // XXX performance test suite.
    // EXPECT_GT(time_exact, time_table);
    // EXPECT_GT(time_cheby, time_table);
    EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// end of file
