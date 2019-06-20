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
 * arbitrary locations.
 *
 * Compared to the TU Chemnitz implementation there are some differences in
 * conventions intended to make things simpler in the SAR context:
 *      -# Sign conventions.  NFFT.execute() uses a positive phase (backward),
 *         while NFFT.execute_adjoint() uses a negative phase (forward).  This
 *         consistent with  uses where the frequency spectrum is uniformly
 *         sampled and the time domain is not.
 *
 *      -# Order of spectra.  Spectra are expected to be in FFTW order, e.g.
 *         low frequencies at the ends and high frequencies in the middle.
 *
 *      -# Scaling of time/frequency.  Locations are specified as sample numbers
 *         (at rate consistent with spectrum size) instead of the unit torus.
 *
 *      -# Sample locations do not need to be specified in advance.  You can
 *         use NFFT.set_spectrum and then NFFT.interp all the points you want
 *         on the fly.  The NFFT.execute convenience function combines these.
 */
template<class T>
class isce::signal::NFFT {
    public:
        /** NFFT Constructor.
         *
         * @param[in] m         Interpolation kernel size parameter (width=2*m+1).
         * @param[in] n         Length of spectrum.
         * @param[in] fft_size  Transform size (> n).
         */
        NFFT(size_t m, size_t n, size_t fft_size);

        /** Execute a transform.
         *
         * Equivalent to \f[ f_j = \frac{1}{N} \sum_{k=-\frac{N}{2}}^{\frac{N}{2}-1} \hat{f}_k \exp(+2\pi i k t_j / N) \f]
         * Where \f$ N \f$ is the size of the spectrum.
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

        /** Execute an adjoint transform.
         *
         * Equivalent to \f[ \hat{f}_k = \sum_{j=0}^{M-1} f_j \exp(-2\pi i k t_j / N) \f]
         * Where \f$ M \f$ is the number of time samples and \f$ N \f$ is the
         * size of the spectrum.
         *
         * @param[in]  time_series  Signal to transform.
         * @param[in]  times        Sample locations in [0:n) of input signal.
         * @param[out] spectrum     Storage for output spectrum.
         *                          Length should be equal to size_spectrum().
         */
        void execute_adjoint(const std::valarray<std::complex<T>> &time_series,
                             const std::valarray<double> &times,
                             std::valarray<std::complex<T>> &spectrum);

        /** Execute an adjoint transform (raw pointer interface).
         *
         * @param[in]  isize        Length of input signal.
         * @param[in]  istride      Stride between elements of input signal.
         * @param[in]  time_series  Signal to transform.
         * @param[in]  tsize        Length of time vector.  Should == isize.
         * @param[in]  tstride      Stride of time vector.
         * @param[in]  times        Sample locations in [0:n) of input signal.
         * @param[in]  osize        Length of output signal (== size_spectrum())
         * @param[in]  ostride      Stride between elements of output signal.
         * @param[out] spectrum     Storage for output spectrum.
         */
        void execute_adjoint(size_t isize, size_t istride,
                             const std::complex<T> *time_series,
                             size_t tsize, size_t tstride,
                             const double *times,
                             size_t osize, size_t ostride,
                             std::complex<T> *spectrum);


        /** Ingest a spectrum for transform.
         *
         * @param[in] x         Spectrum to transform.  Should be in FFTW order,
         *                      e.g., [0:pi, -pi:0).
         *
         * This function will filter, zero-pad, and transform the input data.
         * After this you can call NFFT.interp.
         *
         * @see execute Is an alternative strategy.
         * @see interp
         */
        void set_spectrum(const std::valarray<std::complex<T>> &x);

        /** Ingest a spectrum for transform (raw pointer interface).
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

        /** Interpolate the transformed signal.
         *
         * @param[in] t     Location in [0,n) to sample the time-domain signal.
         *
         * @see execute is an alternative strategy.
         * @see set_spectrum must be called first.
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
