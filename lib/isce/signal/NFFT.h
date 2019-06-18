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
 * It takes a uniformly sampled spectrum and produces time-domain samples at
 * arbitrary locations (here times are not scaled to the unit torus).
 * This implementation differs in that the grid points do not need to be
 * specified ahead of time, which is more convenient for SAR backprojection.
 * Typical usage will entail three steps:
 *   -# Construct NFFT object
 *   -# Feed it regularly-sampled frequency-domain data with `set_spectrum`.
 *   -# Request arbitrary time-domain samples with `interp`.
 * The convenience method execute combines the last two steps into one.
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

        /** Execute a transform.
         *
         * @param[in]  spectrum Signal to transform, in FFTW order.
         * @param[in]  times    Desired sample locations in [0:n)
         * @param[out] out      Storage for output signal, same length as times.
         *
         * Equivalent to set_spectrum and out[i]=NFFT::interp(times[i]).
         * @see set_spectrum
         * @see interp
         */
        void execute(const std::valarray<std::complex<T>> &spectrum,
                     const std::valarray<double> &times,
                     std::valarray<std::complex<T>> &out);

        /** Execute a transform (raw pointer interface).
         *
         * @param[in]  isize    Length of spectrum (should be == n)
         * @param[in]  istride  Stride between elements of spectrum.
         * @param[in]  spectrum Signal to transform, in FFTW order.
         * @param[in]  tsize    Number of output time samples.
         * @param[in]  tstride  Stride between elements of time array.
         * @param[in]  times    Desired sample locations in [0:n)
         * @param[in]  tsize    Number of output samples (should be == tsize).
         * @param[in]  tstride  Stride between elements of output array.
         * @param[out] out      Storage for output signal.
         *
         * Equivalent to set_spectrum and out[i]=NFFT::interp(times[i]).
         * @see set_spectrum
         * @see interp
         */
        void execute(size_t isize, size_t istride,
                     const std::complex<T> *spectrum,
                     size_t tsize, size_t tstride,
                     const double *times,
                     size_t osize, size_t ostride,
                     std::complex<T> *out);

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
