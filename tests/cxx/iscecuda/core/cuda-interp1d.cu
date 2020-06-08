#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include <gtest/gtest.h>

#include <isce/cuda/core/Interp1d.h>
#include <isce/cuda/core/Kernels.h>
#include <isce/cuda/except/Error.h>
#include <isce/except/Error.h>
#include <isce/math/Sinc.h>

using namespace isce::cuda::core;
using namespace isce::except;

using thrust::complex;

// Adapted from julia code at
// https://github.jpl.nasa.gov/bhawkins/FIRInterp.jl/blob/master/test/common.jl
class TestSignal {
public:
    TestSignal(int n, double bw, unsigned seed = 12345) : _bw(bw)
    {
        if (bw < 0. or bw >= 1.) {
            throw DomainError(ISCE_SRCINFO(), "bandwidth must be in [0, 1)");
        }

        int nt = 4 * n;
        _t.resize(nt);
        _w.resize(nt);

        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0., 1. * n / nt);
        std::uniform_real_distribution<double> uniform(0., n - 1.);

        std::generate(_t.begin(), _t.end(), [&]() { return uniform(rng); });
        std::generate(_w.begin(), _w.end(), [&]() {
            return complex<double>(normal(rng), normal(rng));
        });
    }

    std::complex<double> eval(double t)
    {
        complex<double> z = {0., 0.};
        auto n = static_cast<int>(_w.size());
        for (int i = 0; i < n; ++i) {
            z += _w[i] * isce::math::sinc(_bw * (t - _t[i]));
        }
        return z;
    }

    std::vector<complex<double>> eval(const std::vector<double>& times)
    {
        std::vector<complex<double>> z(times.size());
        std::transform(times.begin(), times.end(), z.begin(),
                       [&](double t) { return eval(t); });
        return z;
    }

private:
    double _bw;
    std::vector<double> _t;
    std::vector<complex<double>> _w;
};

template<class Kernel>
__global__ void interp(Kernel kernel, complex<double>* out, const double* times,
                       size_t out_n, const complex<double>* signal, size_t in_n)
{
    auto tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= out_n) {
        return;
    }
    out[tid] = interp1d(kernel, signal, in_n, 1, times[tid]);
}

template<class Kernel>
std::vector<complex<double>>
interpolate(const Kernel& kernel, const std::vector<complex<double>>& signal,
            const std::vector<double>& times)
{
    // copy signal and interp times to the device
    thrust::device_vector<complex<double>> d_signal = signal;
    thrust::device_vector<double> d_times = times;

    // create device vector to store output
    thrust::device_vector<complex<double>> d_out(times.size());

    using KV = typename Kernel::view_type;

    // interpolate signal on the device
    int block = 128;
    int grid = (times.size() + block - 1) / block;
    interp<KV><<<grid, block>>>(kernel, d_out.data().get(),
                                d_times.data().get(), times.size(),
                                d_signal.data().get(), signal.size());

    // check for kernel launch/execution errors
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // copy result to host
    std::vector<complex<double>> out(d_out.size());
    thrust::copy(d_out.begin(), d_out.end(), out.begin());
    return out;
}

template<typename T>
void checkSameSize(const std::vector<T>& x, const std::vector<T>& y)
{
    if (x.size() != y.size()) {
        throw LengthError(ISCE_SRCINFO(),
                          "input vectors must have the same size");
    }
}

// compute the magnitude of the complex correlation between the two input
// signals
double correlation(const std::vector<complex<double>>& x,
                   const std::vector<complex<double>>& y)
{
    checkSameSize(x, y);
    auto n = static_cast<int>(x.size());

    double xy = 0., xx = 0., yy = 0.;
    for (int i = 0; i < n; ++i) {
        auto xi = x[i];
        auto yi = y[i];
        xy += abs(xi * conj(yi));
        xx += (xi * conj(xi)).real();
        yy += (yi * conj(yi)).real();
    }

    return xy / std::sqrt(xx * yy);
}

// return the arithmetic mean of the input vector
double mean(const std::vector<double>& v)
{
    double sum = std::accumulate(v.begin(), v.end(), 0.);
    auto n = static_cast<double>(v.size());
    return sum / n;
}

// estimate the sample standard deviation of the input vector
double stddev(const std::vector<double>& v, int ddof = 1)
{
    double mu = mean(v);

    double sse = 0.;
    std::for_each(v.begin(), v.end(),
                  [&](double d) { sse += (d - mu) * (d - mu); });

    auto n = static_cast<double>(v.size());
    return std::sqrt(sse / (n - ddof));
}

// estimate the standard deviation of the phase difference between the two
// inputs, assuming no phase wrapping
double phaseStddev(const std::vector<complex<double>>& x,
                   const std::vector<complex<double>>& y)
{
    checkSameSize(x, y);
    auto n = static_cast<int>(x.size());

    std::vector<double> phi(n);
    std::transform(x.begin(), x.end(), y.begin(), phi.begin(),
                   [](complex<double> lhs, complex<double> rhs) {
                       return arg(lhs * conj(rhs));
                   });

    return stddev(phi);
}

// convert linear amplitude to decibels
double dB(double x) { return 20. * std::log10(x); }

void powerBiasStddev(double* bias, double* spread,
                     const std::vector<complex<double>>& x,
                     const std::vector<complex<double>>& y, double minval)
{
    checkSameSize(x, y);
    auto n = static_cast<int>(x.size());

    std::vector<double> ratio;
    for (int i = 0; i < n; ++i) {
        auto ax = abs(x[i]);
        auto ay = abs(y[i]);
        if (ax >= minval and ay >= minval) {
            ratio.push_back(ax / ay);
        }
    }

    *bias = dB(mean(ratio));

    auto m = static_cast<int>(ratio.size());
    std::vector<double> dbr(m);
    std::transform(ratio.begin(), ratio.end(), dbr.begin(),
                   [](double r) { return dB(r); });

    *spread = stddev(dbr);
}

// convert radians to degrees
constexpr double rad2deg(double phi) { return phi * 180. / M_PI; }

struct Interp1dTest : public testing::Test {
    // Length of input signal
    const int n = 512;
    // signal bandwidth as a fraction of sample rate
    const double bw = 0.8;
    // Trick to get apples-to-apples comparisons even for different kernel
    // widths.  See assertions for acceptable range of values
    const int pad = 8;
    // Amplitude mask for backscatter
    const double minval = 1e-6;

    // Generator of bandlimited test signal at arbitrary time samples
    TestSignal ts;
    // Realization of signal at integer time steps
    std::vector<complex<double>> signal;

    Interp1dTest() : ts(n, bw)
    {
        assert(pad > 0 and 2 * pad < n);

        // Generate signal at integer sample times.
        std::vector<double> times(n);
        std::iota(times.begin(), times.end(), 0.);
        signal = ts.eval(times);
    }

    template<class Kernel>
    void testFixedOffset(const Kernel& kernel, double off, double min_corr,
                         double max_phs, double max_bias, double max_spread)
    {
        std::printf("Testing fixed offset = %g\n", off);

        std::vector<double> times(n);
        std::iota(times.begin(), times.end(), off);

        checkInterp(kernel, times, min_corr, max_phs, max_bias, max_spread);
    }

    template<class Kernel>
    void testRandomOffsets(const Kernel& kernel, double min_corr,
                           double max_phs, double max_bias, double max_spread,
                           unsigned seed = 12345)
    {
        std::printf("Testing random offsets.\n");

        std::mt19937 rng(2 * seed);
        std::uniform_real_distribution<double> uniform(-0.5, 0.5);

        std::vector<double> times(n);
        for (int i = 0; i < n; ++i) {
            times[i] = i + uniform(rng);
        }

        checkInterp(kernel, times, min_corr, max_phs, max_bias, max_spread);
    }

    template<class Kernel>
    void checkInterp(const Kernel& kernel, const std::vector<double>& times,
                     double min_corr, double max_phs, double max_bias,
                     double max_spread)
    {
        // evaluate signal and interpolate signal at test times
        std::vector<complex<double>> ref = ts.eval(times);
        std::vector<complex<double>> out = interpolate(kernel, signal, times);

        // mask boundary values
        auto nt = static_cast<int>(times.size());
        for (int i = 0; i < nt; ++i) {
            auto t = times[i];
            if (t < pad or t > n - 1 - pad) {
                out[i] = ref[i] = 0.;
            }
        }

        // check results
        auto corr = correlation(ref, out);
        auto phs = rad2deg(phaseStddev(ref, out));
        double bias, spread;
        powerBiasStddev(&bias, &spread, ref, out, minval);

        EXPECT_GE(corr, min_corr);
        EXPECT_LE(phs, max_phs);
        EXPECT_LE(bias, max_bias);
        EXPECT_LE(spread, max_spread);
        std::printf("min_corr   %9.6f | corr   %9.6f\n", min_corr, corr);
        std::printf("max_phs    %9.6f | phs    %9.6f\n", max_phs, phs);
        std::printf("max_bias   %9.6f | bias   %9.6f\n", max_bias, bias);
        std::printf("max_spread %9.6f | spread %9.6f\n", max_spread, spread);
        std::printf("\n");
    }
};

TEST_F(Interp1dTest, Linear)
{
    auto kernel = LinearKernel<double>();

    // offset = 0 should give back original data for this kernel
    testFixedOffset(kernel, 0., 0.999999, 0.001, 0.001, 0.001);

    testFixedOffset(kernel, -0.3, 0.95, 30., 5., 5.);
    testFixedOffset(kernel, 0.3, 0.95, 30., 5., 5.);
    testFixedOffset(kernel, -0.5, 0.95, 40., 5., 5.);
    testFixedOffset(kernel, 0.5, 0.95, 40., 5., 5.);

    testRandomOffsets(kernel, 0.95, 30., 3., 3.);
}

TEST_F(Interp1dTest, Knab)
{
    auto kernel = KnabKernel<double>(9., 0.8);

    // offset = 0 should give back original data for this kernel
    testFixedOffset(kernel, 0., 0.999999, 0.001, 0.001, 0.001);

    testFixedOffset(kernel, -0.3, 0.998, 5., 1., 1.);
    testFixedOffset(kernel, 0.3, 0.998, 5., 1., 1.);
    testFixedOffset(kernel, -0.3, 0.998, 5., 1., 1.);
    testFixedOffset(kernel, 0.5, 0.998, 5., 1., 1.);

    testRandomOffsets(kernel, 0.998, 5., 0.5, 0.5);
}

TEST_F(Interp1dTest, TabulatedKnab)
{
    auto knab = KnabKernel<double>(9., 0.8);
    auto kernel = TabulatedKernel<double>(knab, 2048);

    // offset = 0 should give back original data for this kernel
    testFixedOffset(kernel, 0., 0.999999, 0.001, 0.001, 0.001);

    testRandomOffsets(kernel, 0.998, 5.0, 0.5, 0.5);
}

TEST_F(Interp1dTest, ChebyKnab)
{
    auto knab = KnabKernel<double>(9., 0.8);
    auto kernel = ChebyKernel<double>(knab, 16);

    // offset = 0 should give back original data for this kernel
    testFixedOffset(kernel, 0., 0.999999, 0.001, 0.001, 0.001);

    testRandomOffsets(kernel, 0.998, 5.0, 0.5, 0.5);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
