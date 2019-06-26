//
// Author: Brian Hawkins
// Copyright 2019
//

#include "Kernels.h"
#include <isce/except/Error.h>
#include <complex>
#include <cmath>

using isce::except::RuntimeError;

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
    if (!isfinite(window)) {
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
    _scale = 1.0 / (M_PI * std::cyl_bessel_i(0, _m*_b));
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
        return _scale;
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
