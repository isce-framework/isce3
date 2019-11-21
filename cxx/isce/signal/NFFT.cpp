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
NFFT(size_t m, size_t n, size_t fft_size)
    : _m(m), _n(n), _fft_size(fft_size), _kernel(m,n,fft_size)
{
    if (n >= fft_size) {
        throw LengthError(ISCE_SRCINFO(), "Require N<NFFT for zero-padding.");
    }
    if (n % 2 == 1) {
        throw LengthError(ISCE_SRCINFO(), "Must have even length spectrum.");
    }
    // Allocate arrays
    _xf.resize(fft_size);
    _xt.resize(fft_size);
    _weights.resize(n);

    // Setup inverse FFT.
    int sizes[] = {(int)fft_size};
    _fft.fftPlanBackward(_xf, _xt, /*rank*/1, &sizes[0], /*howmany*/1,
                         /*inembed*/NULL, /*istride*/1, /*idist*/0,
                         /*onembed*/NULL, /*ostride*/1, /*odist*/0,
                         FFTW_BACKWARD);

    // Setup forward FFT for adjoint.
    _fft.fftPlanForward(_xt, _xf, /*rank*/1, &sizes[0], /*howmany*/1,
                        /*inembed*/NULL, /*istride*/1, /*idist*/0,
                        /*onembed*/NULL, /*ostride*/1, /*odist*/0,
                        FFTW_FORWARD);

    // Pre-compute spectral weights (1/phi_hat in NFFT papers).
    // Also include factor of n since FFTW does not normalize DFT.
    T b = M_PI * (2.0 - 1.0*n/fft_size);
    T norm = isce::math::bessel_i0(b*m) / n;
    size_t n2 = (n - 1) / 2 + 1;
    for (size_t i=0; i<n2; ++i) {
        double f = 2 * M_PI * i / _fft_size;
        _weights[i] = norm / isce::math::bessel_i0(m * std::sqrt(b*b-f*f));
    }
    for (size_t i=n2; i<n; ++i) {
        double f = 2 * M_PI * ((double)i - n) / _fft_size;
        _weights[i] = norm / isce::math::bessel_i0(m * std::sqrt(b*b-f*f));
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
    // Clear any old data.
    for (size_t i=0; i<_fft_size; ++i) {
        _xf[i] = 0;
    }
    // Zero-pad and scale spectrum.
    size_t n2 = size / 2;
    for (size_t i=0; i<n2; ++i) {
        _xf[i] = x[i*stride] * _weights[i];
    }
    for (size_t i=n2; i>0; --i) {
        _xf[_fft_size-i] = x[(size-i)*stride] * _weights[size-i];
    }
    // NOTE For even lengths we're not splitting Nyquist bin.
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
interp(double t) const
{
    // scale time index to account for zero-padding of spectrum.
    t *= (double)_fft_size / (double)_n;
    return isce::core::interp1d<T,std::complex<T>>(_kernel, _xt, t,
                                                   /*periodic*/true);
}

template<class T>
void
isce::signal::NFFT<T>::
execute(size_t isize, size_t istride,
        const std::complex<T> *spectrum,
        size_t tsize, size_t tstride,
        const double *times,
        size_t osize, size_t ostride,
        std::complex<T> *out)
{
    set_spectrum(isize, istride, spectrum);
    if (osize < tsize) {
        throw LengthError(ISCE_SRCINFO(), "Insufficient storage");
    }
    for (size_t i=0; i<tsize; ++i) {
        double t = times[i*tstride];
        out[i*ostride] = interp(t);
    }
}

template<class T>
void
isce::signal::NFFT<T>::
execute(const std::valarray<std::complex<T>> &spectrum,
        const std::valarray<double> &times,
        std::valarray<std::complex<T>> &out)
{
    const size_t stride = 1;
    execute(spectrum.size(), stride, &spectrum[0],
            times.size(), stride, &times[0],
            out.size(), stride, &out[0]);
}

// Execute adjoint transform.
template<class T>
void
isce::signal::NFFT<T>::
execute_adjoint(size_t isize, size_t istride,
                const std::complex<T> *time_series,
                size_t tsize, size_t tstride,
                const double *times,
                size_t osize, size_t ostride,
                std::complex<T> *spectrum)
{
    if (osize != _n) {
        throw LengthError(ISCE_SRCINFO(), "Spectrum size != NFFT size.");
    }
    size_t nt = std::min(isize, tsize);  // TODO warn if isize!=tsize?
    // Zero-out data.
    for (size_t i=0; i<_fft_size; ++i) {
        _xt[i] = 0.0;
    }
    // XXX Need signed type for loop over [-m,m].
    const long m = (long) _m;
    // Compute filtered time series.
    for (long i=0; i<nt; ++i) {
        // Input times expected in interval [0,_n).
        // Map time onto unit torus [-0.5,0.5) so zero-padding works correctly.
        double t = times[i*tstride] / _n;
        t = std::fmod(t + 0.5, 1.0) - 0.5;
        // Now scale to padded FFT size.
        t *= _fft_size;
        const long ti = (long)std::round(t);
        const double tf = ti - t;
        for (long j=-m; j<=m; ++j) {
            long k = (ti + j) % (long)_fft_size;
            // XXX Unlike Python, C++ modulo takes sign of dividend.
            if (k < 0) k += (long)_fft_size;
            _xt[k] += _kernel(tf + j) * time_series[i*istride];
        }
    }
    // FFT
    _fft.forward(_xt, _xf);
    // Remove filter response and copy to output.
    const long n2 = _n / 2;
    for (long i=0; i<n2; ++i) {
        spectrum[i*ostride] = _n * _weights[i] * _xf[i];
    }
    for (long i=n2; i>0; --i) {
        spectrum[(_n-i)*ostride] = _n * _weights[_n-i] * _xf[_fft_size-i];
    }
}

template<class T>
void
isce::signal::NFFT<T>::
execute_adjoint(const std::valarray<std::complex<T>> &time_series,
                const std::valarray<double> &times,
                std::valarray<std::complex<T>> &spectrum)
{
    execute_adjoint(time_series.size(), 1, &time_series[0],
                    times.size(), 1, &times[0],
                    spectrum.size(), 1, &spectrum[0]);
}

template class isce::signal::NFFT<float>;
template class isce::signal::NFFT<double>;
