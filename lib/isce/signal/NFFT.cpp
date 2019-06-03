// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Brian Hawkins
// Copyright 2019
//

#include "fftw3cxx.h"  // for FFTW_BACKWARD macro
#include "NFFT.h"
#include <isce/except/Error.h>
#include <isce/core/Interp1d.h>
#include <cmath>

using isce::except::LengthError;

// Constructor
template<class T>
isce::signal::NFFT<T>::
NFFT(size_t m, size_t n, size_t nfft)
    : _m(m), _n(n), _nfft(nfft), _kernel(m,n,nfft)
{
    if (n >= nfft) {
        throw LengthError(ISCE_SRCINFO(), "Require N<NFFT for zero-padding.");
    }
    // Allocate arrays
    _xf.resize(nfft);
    _xt.resize(nfft);
    _weights.resize(n);

    // Setup inverse FFT.
    int sizes[] = {(int)nfft};
    _fft.fftPlanBackward(_xf, _xt, /*rank*/1, &sizes[0], /*howmany*/1,
                         /*inembed*/NULL, /*istride*/1, /*idist*/0,
                         /*onembed*/NULL, /*ostride*/1, /*odist*/0,
                         FFTW_BACKWARD);

    // Pre-compute spectral weights (1/phi_hat in NFFT papers).
    // Also include factor of n since FFTW does not normalize DFT.
    T b = M_PI * (2.0 - 1.0*n/nfft);
    T norm = std::cyl_bessel_i(0, b*m) / n;
    size_t n2 = (n - 1) / 2 + 1;
    for (size_t i=0; i<n2; ++i) {
        double f = 2 * M_PI * i / _nfft;
        _weights[i] = norm / std::cyl_bessel_i(0, m * std::sqrt(b*b-f*f));
    }
    for (size_t i=n2; i<n; ++i) {
        double f = 2 * M_PI * ((double)i - n) / _nfft;
        _weights[i] = norm / std::cyl_bessel_i(0, m * std::sqrt(b*b-f*f));
    }
}

// Digest some data.
template<class T>
void
isce::signal::NFFT<T>::
set_spectrum(size_t size, size_t stride, const std::complex<T> *x)
{
    if (size != _n) {
        throw LengthError(ISCE_SRCINFO(), "Spectrum size != NFFT size.");
    }
    // Zero-pad and scale spectrum.
    size_t n2 = (size - 1) / 2 + 1;
    for (size_t i=0; i<n2; ++i) {
        _xf[i] = x[i*stride] * _weights[i];
    }
    for (size_t i=n2, j=_nfft-n2; i<size; ++i, ++j) {
        _xf[j] = x[i*stride] * _weights[i];
    }
    // Split Nyquist bin correctly.
    if (size % 2 == 0) {
        _xf[n2] = _xf[_nfft-n2] = T(0.5) * x[n2*stride] * _weights[n2];
    }
    // Transform to (expanded) time-domain.
    _fft.inverse(_xf, _xt);
}

// valarray version
template<class T>
void
isce::signal::NFFT<T>::
set_spectrum(const std::valarray<std::complex<T>> &x)
{
    set_spectrum(x.size(), /*stride*/1, &x[0]);
}

// Emit samples.
template<class T>
std::complex<T>
isce::signal::NFFT<T>::
interp(double t)
{
    // scale time index to account for zero-padding of spectrum.
    t *= _nfft / _n;
    return isce::core::interp1d<T,std::complex<T>>(_kernel, _xt, t);
}

template class isce::signal::NFFT<float>;
template class isce::signal::NFFT<double>;
