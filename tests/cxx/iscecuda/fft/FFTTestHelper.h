#pragma once

#include <array>
#include <cmath>
#include <random>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <vector>

/** Copy device vector to host */
template<typename T>
inline
std::vector<T> copyToHost(const thrust::device_vector<T> & d)
{
    std::vector<T> h(d.size());

    if (d.size()) {
        T * dst = h.data();
        const T * src = d.data().get();
        std::size_t count = d.size() * sizeof(T);
        checkCudaErrors( cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost) );
    }

    return h;
}

/** Copy host vector to device */
template<typename T>
inline
thrust::device_vector<T> copyToDevice(const std::vector<T> & h)
{
    thrust::device_vector<T> d(h.size());

    if (h.size()) {
        T * dst = d.data().get();
        const T * src = h.data();
        std::size_t count = h.size() * sizeof(T);
        checkCudaErrors( cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice) );
    }

    return d;
}

/** Absolute value (real-valued input) */
template<typename T>
inline
T abs(T x) { return std::abs(x); }

/** Absolute value (complex-valued input) */
template<typename T>
inline
T abs(const thrust::complex<T> & z) { return thrust::abs(z); }

/** Check if two vectors are element-wise equivalent to within errtol */
template<typename T>
inline
bool compareVectors(const std::vector<T> & lhs, const std::vector<T> & rhs, double errtol)
{
    if (lhs.size() != rhs.size()) { return false; }

    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (::abs(lhs[i] - rhs[i]) >= errtol) { return false; }
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

    thrust::complex<T> sample() { return {_dis(_gen), _dis(_gen)}; }

private:
    std::uniform_real_distribution<T> _dis;
    std::mt19937 _gen;
};

/** 1-D discrete forward Fourier transform with complex-valued input */
template<typename T>
inline
void fwd_dft_c2c_1d(thrust::complex<T> * out,
                    const thrust::complex<T> * in,
                    int n,
                    int stride = 1,
                    int batch = 1,
                    int dist = 0)
{
    thrust::complex<T> j(0., 1.);

    for (int i = 0; i < batch; ++i) {
        for (int k = 0; k < n; ++k) {
            auto & X = out[i * dist + k * stride];
            X = 0.;
            for (int m = 0; m < n; ++m) {
                auto x = in[i * dist + m * stride];
                X += x * thrust::exp(-2*M_PI*j * (m * k / T(n)));
            }
        }
    }
}

/** 1-D discrete inverse Fourier transform with complex-valued output */
template<typename T>
inline
void inv_dft_c2c_1d(thrust::complex<T> * out,
                    const thrust::complex<T> * in,
                    int n,
                    int stride = 1,
                    int batch = 1,
                    int dist = 0)
{
    thrust::complex<T> j(0., 1.);

    for (int i = 0; i < batch; ++i) {
        for (int m = 0; m < n; ++m) {
            auto & x = out[i * dist + m * stride];
            x = 0.;
            for (int k = 0; k < n; ++k) {
                auto X = in[i * dist + k * stride];
                x += X * thrust::exp(2*M_PI*j * (m * k / T(n)));
            }
        }
    }
}

/** 1-D discrete forward Fourier transform with real-valued input */
template<typename T>
inline
void fwd_dft_r2c_1d(thrust::complex<T> * out,
                    const T * in,
                    int n,
                    int stride = 1,
                    int batch = 1,
                    int dist = 0)
{
    thrust::complex<T> j(0., 1.);

    for (int i = 0; i < batch; ++i) {
        for (int k = 0; k < n/2 + 1; ++k) {
            auto & X = out[i * dist + k * stride];
            X = 0.;
            for (int m = 0; m < n; ++m) {
                auto x = in[i * dist + m * stride];
                X += x * thrust::exp(-2*M_PI*j * (m * k / T(n)));
            }
        }
    }
}

/** 1-D discrete inverse Fourier transform with real-valued output */
template<typename T>
inline
void inv_dft_c2r_1d(T * out,
                    const thrust::complex<T> * in,
                    int n,
                    int stride = 1,
                    int batch = 1,
                    int dist = 0)
{
    thrust::complex<T> j(0., 1.);

    for (int i = 0; i < batch; ++i) {
        for (int m = 0; m < n; ++m) {
            thrust::complex<T> x = 0.;
            for (int k = 0; k < n/2 + 1; ++k) {
                auto X = in[i * dist + k * stride];
                x += X * thrust::exp(2*M_PI*j * (m * k / T(n)));
            }
            for (int k = n/2 + 1; k < n; ++k) {
                auto X = thrust::conj(in[i * dist + (n-k) * stride]);
                x += X * thrust::exp(2*M_PI*j * (m * k / T(n)));
            }
            out[i * dist + m * stride] = x.real();
        }
    }
}

/** 2-D discrete forward Fourier transform with complex-valued input */
template<typename T>
inline
void fwd_dft_c2c_2d(thrust::complex<T> * out, const thrust::complex<T> * in, const std::array<int, 2> & n)
{
    std::vector<thrust::complex<T>> z(n[0] * n[1]);
    fwd_dft_c2c_1d(z.data(), in, n[0], 1, n[1], n[0]);
    fwd_dft_c2c_1d(out, z.data(), n[1], n[0], n[0], 1);
}

/** 2-D discrete inverse Fourier transform with complex-valued output */
template<typename T>
inline
void inv_dft_c2c_2d(thrust::complex<T> * out, const thrust::complex<T> * in, const std::array<int, 2> & n)
{
    std::vector<thrust::complex<T>> z(n[0] * n[1]);
    inv_dft_c2c_1d(z.data(), in, n[0], 1, n[1], n[0]);
    inv_dft_c2c_1d(out, z.data(), n[1], n[0], n[0], 1);
}

/** 2-D discrete forward Fourier transform with real-valued input */
template<typename T>
inline
void fwd_dft_r2c_2d(thrust::complex<T> * out, const T * in, const std::array<int, 2> & n)
{
    std::vector<thrust::complex<T>> z(n[0] * n[1]);
    fwd_dft_r2c_1d(z.data(), in, n[0], 1, n[1], n[0]);
    fwd_dft_c2c_1d(out, z.data(), n[1], n[0], n[0], 1);
}

/** 2-D discrete inverse Fourier transform with real-valued output */
template<typename T>
inline
void inv_dft_c2r_2d(T * out, const thrust::complex<T> * in, const std::array<int, 2> & n)
{
    std::vector<thrust::complex<T>> z(n[0] * n[1]);
    inv_dft_c2c_1d(z.data(), in, n[1], n[0], n[0], 1);
    inv_dft_c2r_1d(out, z.data(), n[0], 1, n[1], n[0]);
}
