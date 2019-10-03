//
// Author: Brian Hawkins
// Copyright 2019
//

#include "Kernels.h"

#include <isce/except/Error.h>
#include <complex>
#include <cmath>
#include <type_traits>

using isce::except::RuntimeError;
using isce::except::LengthError;

/* 
 * Bartlett
 */
 
// constructor
template <typename T>
isce::core::BartlettKernel<T>::
BartlettKernel(double width) {
    this->_halfwidth = fabs(width / 2);
}

// call
template <typename T>
T
isce::core::BartlettKernel<T>::
operator()(double t) const {
    double t2 = fabs(t / this->_halfwidth);
    if (t2 > 1.0) {
        return T(0.0);
    }
    return T(1.0 - t2);
}

template class isce::core::BartlettKernel<float>;
template class isce::core::BartlettKernel<double>;

/* 
 * Knab, sampling window from 1983 paper
 */
 
template <typename T>
T
isce::core::sinc(T t)
{
    static T const eps1 = std::sqrt(std::numeric_limits<T>::epsilon());
    static T const eps2 = std::sqrt(eps1);
    T x = M_PI * fabs(t);
    if (x >= eps2) {
        return sin(x) / x;
    } else {
        T out = static_cast<T>(1);
        if (x > eps1) {
            out -= x * x / 6;
        }
        return out;
    }
}

template <typename T> 
T
_sampling_window(T t, T halfwidth, T bandwidth)
{
    if (!((0.0 < bandwidth) && (bandwidth < 1.0))) {
        throw RuntimeError(ISCE_SRCINFO(), "Require 0 < bandwidth < 1");
    }
    const T c = M_PI * halfwidth * (1.0 - bandwidth);
    const T tf = t / halfwidth;
    std::complex<T> y = sqrt((std::complex<T>)(1.0 - tf*tf));
    T window = real(cosh(c * y) / cosh(c));
    if (!std::isfinite(window)) {
        throw RuntimeError(ISCE_SRCINFO(), "Invalid window parameters.");
    }
    return window;
}

// constructor
template <typename T>
isce::core::KnabKernel<T>::
KnabKernel(double width, double bandwidth) {
    this->_halfwidth = fabs(width / 2);
    this->_bandwidth = bandwidth;
}

// call
template <typename T>
T
isce::core::KnabKernel<T>::
operator()(double t) const {
    auto st = isce::core::sinc<T>(t);
    return _sampling_window(t, this->_halfwidth, this->_bandwidth) * st;
}

template class isce::core::KnabKernel<float>;
template class isce::core::KnabKernel<double>;
template double isce::core::sinc(double);

/* 
 * NFFT
 */
 
// constructor
template <typename T>
isce::core::NFFTKernel<T>::
NFFTKernel(size_t m, size_t n, size_t fft_size)
    : _m(m), _n(n), _fft_size(fft_size)
{
    _b = M_PI * (2.0 - 1.0*n/fft_size);
    _scale = 1.0 / (M_PI * isce::math::bessel_i0(_m*_b));
    this->_halfwidth = fabs((2*m+1) / 2.0);
}

// call
template <typename T>
T
isce::core::NFFTKernel<T>::
operator()(double t) const
{
    T x2 = t*t - _m*_m;
    // x=0
    if (std::abs(x2) < std::numeric_limits<double>::epsilon()) {
        return _b * _scale;
    }
    T out = 1.0;
    if (x2 < 0.0) {
        T x = std::sqrt(std::abs(x2));
        out = std::sinh(_b*x) / x;
    } else {
        T x = std::sqrt(x2);
        out = std::sin(_b*x) / x;
    }
    return _scale * out;
}

template class isce::core::NFFTKernel<float>;
template class isce::core::NFFTKernel<double>;

/*
 * Tabulated kernel.
 */

// Constructor
template <typename T>
isce::core::TabulatedKernel<T>::
TabulatedKernel(const isce::core::Kernel<T> &kernel, size_t n)
{
    this->_halfwidth = kernel.width() / 2.0;
    // Need at least two points for linear interpolation.
    if (n < 2) {
        throw LengthError(ISCE_SRCINFO(), "Require table size >= 2.");
    }
    // Need i+1 < n so linear interp doesn't run off of table.
    _imax = n - 2;
    // Allocate table.
    _table.resize(n);
    // Assume Kernel is even and fill table with f(x) for 0 <= x <= halfwidth.
    const double dx = this->_halfwidth / (n - 1.0);
    _1_dx = 1.0 / dx;
    for (size_t i=0; i<n; ++i) {
        double x = i * dx;
        _table[i] = kernel(x);
    }
}

// call
template <typename T>
T
isce::core::TabulatedKernel<T>::
operator()(double x) const
{
    // Return zero outside table.
    auto ax = std::abs(x);
    if (ax > this->_halfwidth) {
        return T(0);
    }
    // Normalize to table sample index.
    auto axn = ax * _1_dx;
    // Determine left side of interval.
    size_t i = std::floor(axn);
    // Make sure floating point multiply doesn't cause off-by-one.
    i = std::min(i, _imax);
    // Linear interpolation.
    return _table[i] + (axn - i) * (_table[i+1] - _table[i]);
}

template class isce::core::TabulatedKernel<float>;
template class isce::core::TabulatedKernel<double>;

template <typename T>
isce::core::ChebyKernel<T>::
ChebyKernel(const isce::core::Kernel<T> &kernel, size_t n)
{
    this->_halfwidth = kernel.width() / 2.0;
    // Fit a kernel with DCT of fn at Chebyshev zeros.
    // Assume even function and fit on interval [0,width/2] to avoid a bunch
    // of zero coefficients.
    std::valarray<T> q(n), fx(n);
    _scale = 4.0 / kernel.width();
    for (long i=0; i<n; ++i) {
        q[i] = M_PI * (2.0*i + 1.0) / (2.0*n);
        // shift & scale [-1,1] to [0,width/2].
        T x = (std::cos(q[i]) + 1.0) / _scale;
        fx[i] = kernel(x);
    }
    // FFTW provides DCT with plan_r2r(REDFT10) but this isn't exposed in
    // isce::core::signal.  Typically we're only fitting a few coefficients
    // anyway so just implement O(n^2) algorithm.
    _coeffs.resize(n);
    for (long i=0; i<n; ++i) {
        _coeffs[i] = 0.0;
        for (long j=0; j<n; ++j) {
            T w = std::cos(i * q[j]);
            _coeffs[i] += w * fx[j];
        }
        _coeffs[i] *= 2.0 / n;
    }
    _coeffs[0] *= 0.5;
}

template <typename T>
T
isce::core::ChebyKernel<T>::
operator()(double x) const
{
    // Careful to avoid weird stuff outside [-1,1] definition.
    const auto ax = std::abs(x);
    if (ax > this->_halfwidth) {
        return T(0);
    }
    // Map [0,L/2] to [-1,1]
    const T q = (ax * _scale) - T(1);
    const T twoq = T(2) * q;
    const int n = _coeffs.size();
    // Clenshaw algorithm for two term recurrence.
    T bk=0, bk1=0, bk2=0;
    for (int i=n-1; i>0; --i) {
        bk = _coeffs[i] + twoq * bk1 - bk2;
        bk2 = bk1;
        bk1 = bk;
    }
    return _coeffs[0] + q*bk1 - bk2;
}

template class isce::core::ChebyKernel<float>;
template class isce::core::ChebyKernel<double>;
