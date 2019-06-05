// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Brian Hawkins
// Copyright 2019-
//

#ifndef ISCE_SIGNAL_NFFT_H
#define ISCE_SIGNAL_NFFT_H

#include <cmath>
#include <valarray>

#include <isce/core/Constants.h>
#include <isce/core/Kernels.h>
#include "Signal.h"

// Declaration
namespace isce {
    namespace signal {
        template<class T> class NFFT;
    }
}

/** Non-equispaced fast Fourier transform (NFFT)
 *
 * Class implementing NFFT algorithm described in @cite keiner2009 .
 * This implementation differs in that the grid points do not need to be
 * specified ahead of time, which is more convenient for SAR backprojection.
 * Typical usage will entail three steps:
 *   -# Construct NFFT object
 *   -# Feed it regularly-sampled frequency-domain data with `set_spectrum`.
 *   -# Request arbitrary time-domain samples with `interp`.
 */
template<class T>
class isce::signal::NFFT {
    public:
        /** NFFT Constructor.
         *
         * @param[in] m         Interpolation kernel size parameter (width=2*m+1).
         * @param[in] n         Length of input spectrum.
         * @param[in] fft_size  Transform size (> n).
         */
        NFFT(size_t m, size_t n, size_t fft_size);

        /** Ingest a spectrum for transform.
         *
         * @param[in] size      Length of signal.  Should be same as n in
         *                      constructor or else zero-padding will be wrong.
         * @param[in] stride    Stride between array elements.
         * @param[in] x         Spectrum to transform.  Should be in FFTW order,
         *                      e.g., [0:pi, -pi:0).
         *
         * This function will filter, zero-pad, and transform the input data.
         * After this you can call the interp method.
         */
        void set_spectrum(size_t size, size_t stride, const std::complex<T> *x);

        /** Ingest a spectrum for transform.
         *
         * @param[in] x         Spectrum to transform.  Should be in FFTW order,
         *                      e.g., [0:pi, -pi:0).
         *
         * This function will filter, zero-pad, and transform the input data.
         * After this you can call the interp method.
         */
        void set_spectrum(const std::valarray<std::complex<T>> &x);

        /** Interpolate the transformed signal.
         *
         * @param[in] t     Location in [0,n) to sample the time-domain signal.
         */
        std::complex<T> interp(double t) const;

        size_t size_kernel() const {return 2*_m+1;}
        size_t size_spectrum() const {return _n;}
        size_t size_transform() const {return _fft_size;}

    private:
        size_t _m, _n, _fft_size;
        std::valarray<std::complex<T>> _xf, _xt;
        std::valarray<T> _weights;
        isce::core::NFFTKernel<T> _kernel;
        isce::signal::Signal<T> _fft;
};

#endif
