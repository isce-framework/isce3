#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

/** Check if two vectors are element-wise equivalent to within errtol */
template<typename T>
inline
bool compareVectors(const std::vector<T> & lhs, const std::vector<T> & rhs, double errtol)
{
    if (lhs.size() != rhs.size()) { return false; }

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (std::abs(lhs[i] - rhs[i]) >= errtol) { return false; }
    }

    return true;
}

/** Helper class for sampling from a real-valued uniform distribution */
template<typename T>
class RealUniformDistribution {
public:

    RealUniformDistribution(T low, T high, unsigned int seed = 1234) :
        _dis(low, high),
        _gen(seed)
    {}

    T sample() { return _dis(_gen); }

private:
    std::uniform_real_distribution<T> _dis;
    std::mt19937 _gen;
};

/** Helper class for sampling from a complex-valued uniform distribution (w/ i.i.d. real, imag components) */
template<typename T>
class ComplexUniformDistribution {
public:

    ComplexUniformDistribution(T low, T high, unsigned int seed = 1234) :
        _dis(low, high),
        _gen(seed)
    {}

    std::complex<T> sample() { return {_dis(_gen), _dis(_gen)}; }

private:
    std::uniform_real_distribution<T> _dis;
    std::mt19937 _gen;
};

/** 1-D discrete forward Fourier transform with complex-valued input */
template<typename T>
inline
void fwd_dft_c2c_1d(std::complex<T> * out,
                    const std::complex<T> * in,
                    int n,
                    int stride = 1,
                    int batch = 1,
                    int dist = 0)
{
    constexpr std::complex<T> j(0., 1.);

    for (int i = 0; i < batch; ++i) {
        for (int k = 0; k < n; ++k) {
            auto & X = out[i * dist + k * stride];
            X = 0.;
            for (int m = 0; m < n; ++m) {
                auto x = in[i * dist + m * stride];
                X += x * std::exp(-2*M_PI*j * (m * k / T(n)));
            }
        }
    }
}

/** 1-D discrete inverse Fourier transform with complex-valued output */
template<typename T>
inline
void inv_dft_c2c_1d(std::complex<T> * out,
                    const std::complex<T> * in,
                    int n,
                    int stride = 1,
                    int batch = 1,
                    int dist = 0)
{
    constexpr std::complex<T> j(0., 1.);

    for (int i = 0; i < batch; ++i) {
        for (int m = 0; m < n; ++m) {
            auto & x = out[i * dist + m * stride];
            x = 0.;
            for (int k = 0; k < n; ++k) {
                auto X = in[i * dist + k * stride];
                x += X * std::exp(2*M_PI*j * (m * k / T(n)));
            }
        }
    }
}

/** 1-D discrete forward Fourier transform with real-valued input */
template<typename T>
inline
void fwd_dft_r2c_1d(std::complex<T> * out,
                    const T * in,
                    int n,
                    int stride = 1,
                    int batch = 1,
                    int dist = 0)
{
    constexpr std::complex<T> j(0., 1.);

    for (int i = 0; i < batch; ++i) {
        for (int k = 0; k < n/2 + 1; ++k) {
            auto & X = out[i * dist + k * stride];
            X = 0.;
            for (int m = 0; m < n; ++m) {
                auto x = in[i * dist + m * stride];
                X += x * std::exp(-2*M_PI*j * (m * k / T(n)));
            }
        }
    }
}

/** 1-D discrete inverse Fourier transform with real-valued output */
template<typename T>
inline
void inv_dft_c2r_1d(T * out,
                    const std::complex<T> * in,
                    int n,
                    int stride = 1,
                    int batch = 1,
                    int dist = 0)
{
    constexpr std::complex<T> j(0., 1.);

    for (int i = 0; i < batch; ++i) {
        for (int m = 0; m < n; ++m) {
            std::complex<T> x = 0.;
            for (int k = 0; k < n/2 + 1; ++k) {
                auto X = in[i * dist + k * stride];
                x += X * std::exp(2*M_PI*j * (m * k / T(n)));
            }
            for (int k = n/2 + 1; k < n; ++k) {
                auto X = std::conj(in[i * dist + (n-k) * stride]);
                x += X * std::exp(2*M_PI*j * (m * k / T(n)));
            }
            out[i * dist + m * stride] = x.real();
        }
    }
}

/** 2-D discrete forward Fourier transform with complex-valued input */
template<typename T>
inline
void fwd_dft_c2c_2d(std::complex<T> * out, const std::complex<T> * in, const std::array<int, 2> & n)
{
    std::vector<std::complex<T>> z(n[0] * n[1]);
    fwd_dft_c2c_1d(z.data(), in, n[0], 1, n[1], n[0]);
    fwd_dft_c2c_1d(out, z.data(), n[1], n[0], n[0], 1);
}

/** 2-D discrete inverse Fourier transform with complex-valued output */
template<typename T>
inline
void inv_dft_c2c_2d(std::complex<T> * out, const std::complex<T> * in, const std::array<int, 2> & n)
{
    std::vector<std::complex<T>> z(n[0] * n[1]);
    inv_dft_c2c_1d(z.data(), in, n[0], 1, n[1], n[0]);
    inv_dft_c2c_1d(out, z.data(), n[1], n[0], n[0], 1);
}

/** 2-D discrete forward Fourier transform with real-valued input */
template<typename T>
inline
void fwd_dft_r2c_2d(std::complex<T> * out, const T * in, const std::array<int, 2> & n)
{
    std::vector<std::complex<T>> z(n[0] * n[1]);
    fwd_dft_r2c_1d(z.data(), in, n[0], 1, n[1], n[0]);
    fwd_dft_c2c_1d(out, z.data(), n[1], n[0], n[0], 1);
}

/** 2-D discrete inverse Fourier transform with real-valued output */
template<typename T>
inline
void inv_dft_c2r_2d(T * out, const std::complex<T> * in, const std::array<int, 2> & n)
{
    std::vector<std::complex<T>> z(n[0] * n[1]);
    inv_dft_c2c_1d(z.data(), in, n[1], n[0], n[0], 1);
    inv_dft_c2r_1d(out, z.data(), n[0], 1, n[1], n[0]);
}
