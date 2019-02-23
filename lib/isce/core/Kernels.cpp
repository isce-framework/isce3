//
// Author: Brian Hawkins
// Copyright 2019
//

#include "Kernels.h"
#include <assert.h>
#include <complex>
#include <cmath>

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
operator()(double t) {
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
_sinc(T t)
{
    static T const eps1 = sqrt(std::numeric_limits<T>::epsilon());
    static T const eps2 = sqrt(eps1);
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
    // std::cout << "bandwidth = " << bandwidth << std::endl;
    assert((0.0 < bandwidth) && (bandwidth < 1.0));
    const T c = M_PI * halfwidth * (1.0 - bandwidth);
    const T tf = t / halfwidth;
    std::complex<T> y = sqrt((std::complex<T>)(1.0 - tf*tf));
    T window = real(cosh(c * y) / cosh(c));
    assert(isfinite(window));
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
operator()(double t) {
    return _sampling_window(t, this->_halfwidth, this->_bandwidth) * _sinc(t);
}

template class isce::core::KnabKernel<float>;
template class isce::core::KnabKernel<double>;

