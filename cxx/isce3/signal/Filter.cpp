// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#include "Filter.h"
#include <pyre/journal.h>
#include <isce3/math/complexOperations.h>
using namespace isce3::math::complex_operations;

/**
 * @param[in] signal a block of data to filter
 * @param[in] spectrum a block of spectrum, which is internally used for FFT
 * computations
 * @param[in] ncols number of columns of the block of the data
 * @param[in] nrows number of rows of the block of the data
 */

template <class T>
void
isce3::signal::Filter<T>::
initiateRangeFilter(std::valarray<std::complex<T>> &signal,
                    std::valarray<std::complex<T>> &spectrum,
                    size_t ncols,
                    size_t nrows)
{
    _signal.forwardRangeFFT(signal, spectrum, ncols, nrows);
    _signal.inverseRangeFFT(spectrum, signal, ncols, nrows);
}

/**
 * @param[in] signal a block of data to filter
 * @param[in] spectrum a block of spectrum, which is internally used for FFT
 * computations
 * @param[in] ncols number of columns of the block of the data
 * @param[in] nrows number of rows of the block of the data
 */
template <class T>
void
isce3::signal::Filter<T>::
initiateAzimuthFilter(std::valarray<std::complex<T>> &signal,
                    std::valarray<std::complex<T>> &spectrum,
                    size_t ncols,
                    size_t nrows)
{
    _signal.forwardAzimuthFFT(signal, spectrum, ncols, nrows);
    _signal.inverseAzimuthFFT(spectrum, signal, ncols, nrows);
}

/**
 * @param[in] rangeSamplingFrequency range sampling frequency, in Hz
 * @param[in] subBandCenterFrequencies a vector of center frequencies for each band, in Hz
 * @param[in] subBandBandwidths a vector of bandwidths for each band, in Hz
 * @param[in] signal a block of data to filter
 * @param[in] spectrum a block of spectrum, which is internally used for FFT computations
 * @param[in] ncols number of columns of the block of data
 * @param[in] nrows number of rows of the block of data
 * @param[in] filterType type of the band-pass filter
 */
template <class T>
void
isce3::signal::Filter<T>::
constructRangeBandpassFilter(double rangeSamplingFrequency,
                                std::valarray<double> subBandCenterFrequencies,
                                std::valarray<double> subBandBandwidths,
                                std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows,
                                std::string filterType)
{
    _signal.forwardRangeFFT(signal, spectrum, ncols, nrows);
    _signal.inverseRangeFFT(spectrum, signal, ncols, nrows);

    constructRangeBandpassFilter(rangeSamplingFrequency,
                                subBandCenterFrequencies,
                                subBandBandwidths,
                                ncols,
                                nrows,
                                filterType);
}

template <class T>
void
isce3::signal::Filter<T>::
constructRangeBandpassFilter(double rangeSamplingFrequency,
                                std::valarray<double> subBandCenterFrequencies,
                                std::valarray<double> subBandBandwidths,
                                size_t ncols,
                                size_t nrows,
                                std::string filterType)
{

    int fft_size = ncols;

    _filter.resize(fft_size*nrows);
    std::valarray<std::complex<T>> _filter1D(fft_size); //
    _filter1D = std::complex<T>(0.0,0.0);

    std::valarray<double> frequency(fft_size);
    double dt = 1.0/rangeSamplingFrequency;
    fftfreq(dt, frequency);

    if (filterType=="boxcar"){
        constructRangeBandpassBoxcar(
                            subBandCenterFrequencies,
                            subBandBandwidths,
                            dt,
                            fft_size,
                            _filter1D);

    } else if (filterType=="cosine"){
        double beta = 0.25;
        constructRangeBandpassCosine(subBandCenterFrequencies,
                            subBandBandwidths,
                            dt,
                            frequency,
                            beta,
                            _filter1D);
    } else if (filterType=="kaiser"){
        double beta = 1.6;
        constructRangeBandpassKaiser(subBandCenterFrequencies,
                            subBandBandwidths,
                            dt,
                            fft_size,
                            frequency,
                            beta,
                            _filter1D);
    } else {
        std::string err_str = filterType + " filter has not been implemented";
        pyre::journal::error_t err(
            "isce.signal.Filter.constructRangeBandpassFilter");
        err << err_str << pyre::journal::endl;
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
            err_str);
    }

    //construct a block of the filter with normalization
    const std::complex<T> norm(fft_size, fft_size);
    for (size_t line = 0; line < nrows; line++ ){
        for (size_t col = 0; col < fft_size; col++ ){
            _filter[line*fft_size+col] = _filter1D[col] / norm;
        }
    }
}

/**
 * @param[in] rangeSamplingFrequency range sampling frequency, in Hz
 * @param[in] subBandCenterFrequency center frequency, in Hz
 * @param[in] subBandBandwidth bandwidth, in Hz
 * @param[in, out] signal a block of data to filter
 * @param[in, out] spectrum a block of spectrum, which is internally used for FFT computations
 * @param[in] ncols number of columns of the block of data
 * @param[in] nrows number of rows of the block of data
 * @param[in] filterType type of the band-pass filter
 * @param[in] windowParameter filter parameter
 * @param[in] maxFilterKernelSize maximum kernel size of the FIR filter in time domain

 */
template <class T>
void
isce3::signal::Filter<T>::constructRangeCommonBandFilter(const double rangeSamplingFrequency,
                                        const double subBandCenterFrequency,
                                        const double subBandBandwidth,
                                        size_t ncols,
                                        size_t nrows,
                                        const std::string& filterType,
                                        const double windowParameter,
                                        const int maxFilterKernelSize)
{

    int fft_size = ncols;

    _filter.resize(fft_size*nrows);
    std::valarray<std::complex<T>> _filter1D(fft_size); //
    _filter1D = std::complex<T>(0.0,0.0);

    if (filterType=="kaiser"){
        constructRangeCommonBandKaiserFilter(subBandCenterFrequency,
                            subBandBandwidth,
                            rangeSamplingFrequency,
                            fft_size,
                            windowParameter,
                            _filter1D,
                            maxFilterKernelSize);
    } else {
        pyre::journal::error_t err(
                "isce.signal.Filter.constructRangeCommonBandFilter");
        err << filterType <<" filter has not been implemented"
              << pyre::journal::endl;
    }

    // construct a block of the filter, and normalize the transform to recovery the
    // input signal
    #pragma omp parallel for
    for (size_t line = 0; line < nrows; line++ ){
        for (size_t col = 0; col < fft_size; col++ ){
            _filter[line*fft_size+col] = _filter1D[col] / static_cast<T>(fft_size);
        }
    }
}

/**
 * @param[in] subBandCenterFrequencies a vector of center frequencies for each band
 * @param[in] subBandBandwidths a vector of bandwidths for each band
 * @param[in] dt time interval of the signal
 * @param[in] fft_size length of the spectrum
 * @param[out] _filter1D one dimensional boxcar bandpass filter in frequency domain
 */
template <class T>
void
isce3::signal::Filter<T>::
constructRangeBandpassBoxcar(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             int fft_size,
                             std::valarray<std::complex<T>>& _filter1D)
{
    // construct a boxcar bandpass filter in frequency domain
    // which may have several bands defined by centerferquencies and
    // subBandBandwidths
    for (size_t i = 0; i<subBandCenterFrequencies.size(); ++i){
        std::cout << "i: " << i << std::endl;
        //frequency of the lower bound of this band
        double fL = subBandCenterFrequencies[i] - subBandBandwidths[i]/2;

        //frequency of the higher bound of this band
        double fH = subBandCenterFrequencies[i] + subBandBandwidths[i]/2;

        //index of frequencies for fL and fH
        int indL;
        indexOfFrequency(dt, fft_size, fL, indL);
        int indH;
        indexOfFrequency(dt, fft_size, fH, indH);
        std::cout << "fL: "<< fL << " , fH: " << fH << " indL: " << indL << " , indH: " << indH << std::endl;
        if (fL<0 && fH>=0){
            for (size_t ind = indL; ind < fft_size; ++ind){
                _filter1D[ind] = std::complex<T>(1.0, 0.0);
            }
            for (size_t ind = 0; ind < indH; ++ind){
                _filter1D[ind] = std::complex<T>(1.0, 0.0);
            }

        }else{
            for (size_t ind = indL; ind < indH; ++ind){
                _filter1D[ind] = std::complex<T>(1.0, 0.0);
            }
        }
    }

}

/**
 * @param[in] subBandCenterFrequencies a vector of center frequencies for each band
 * @param[in] subBandBandwidths a vector of bandwidths for each band
 * @param[in] dt time interval of the signal
 * @param[in] frequency a vector of frequencies
 * @param[in] beta parameter for the raised cosine filter (0 <= beta <= 1)
 * @param[out] _filter1D one dimensional boxcar bandpass filter in frequency domain
 */
template <class T>
void
isce3::signal::Filter<T>::
constructRangeBandpassCosine(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double,
                             std::valarray<double>& frequency,
                             double beta,
                             std::valarray<std::complex<T>>& _filter1D)
{

    const double norm = 1.0;

    for (size_t i = 0; i<subBandCenterFrequencies.size(); ++i){
        double fmid = subBandCenterFrequencies[i];
        double bandwidth = subBandBandwidths[i];
        const double df = 0.5 * bandwidth * beta;
        for (size_t i = 0; i < frequency.size(); ++i) {

            // Get the absolute value of shifted frequency
            const double freq = std::abs(frequency[i] - fmid);

            // Passband
            if (freq <= (0.5 * bandwidth - df)) {
                _filter1D[i] = std::complex<T>(norm, 0.0);

            // Transition region
            } else if (freq > (0.5 * bandwidth - df) && freq <= (0.5 * bandwidth + df)) {
                _filter1D[i] = std::complex<T>(norm * 0.5 *
                                    (1.0 + std::cos(M_PI / (bandwidth*beta) *
                                    (freq - 0.5 * (1.0 - beta) * bandwidth))), 0.0);

            }
        }

    }
}

/**
 * @param[in] subBandCenterFrequency the center frequency, in Hz
 * @param[in] subBandBandwidth the bandwidth, in Hz
 * @param[in] rangeSamplingFrequency sampling rate of the signal
 * @param[in] fft_size length of the spectrum
 * @param[in] beta parameter for the kaiser filter
 * @param[out] filter1D one dimensional kaiser filter in frequency domain
 * @param[in] maxFilterKernelSize maximum kernel size of the FIR in the time domain
 */
template <class T>
void
isce3::signal::Filter<T>::
constructRangeCommonBandKaiserFilter(const double subBandCenterFrequency,
                             const double subBandBandwidth,
                             const double rangeSamplingFrequency,
                             const int fft_size,
                             const double beta,
                             std::valarray<std::complex<T>>& filter1D,
                             const int maxFilterKernelSize)
{
    if (filter1D.size() <= 0) filter1D.resize(fft_size);
    filter1D = std::complex<T>(0.0, 0.0);

    isce3::signal::Signal<T> signal;
    signal.forwardRangeFFT(filter1D, filter1D, fft_size, 1);

    std::valarray<std::complex<T>> kaiser;
    _design_shaped_lowpass_filter(subBandBandwidth, rangeSamplingFrequency, beta, kaiser);

    const int sizeOfKaiser = kaiser.size();
    int halfSizeOfKaiser = (sizeOfKaiser - 1)/2;

    if (maxFilterKernelSize < sizeOfKaiser) {
        pyre::journal::error_t err(
                "isce.signal.Filter.constructRangeCommonBandKaiserFilter");
        err << "kaiser kernel size " << sizeOfKaiser <<" is greater han maximum kernel size"
              << maxFilterKernelSize
              << pyre::journal::endl;
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                        "kaiser kernel size is greater than maximum kernel size");
    }

    // Zero padding the filter in the middle
    for (size_t ind = halfSizeOfKaiser; ind < sizeOfKaiser; ind ++) {
        filter1D[ind - halfSizeOfKaiser] = kaiser[ind];
    }
    for (size_t ind = 0; ind < halfSizeOfKaiser; ind ++) {
        filter1D[fft_size - halfSizeOfKaiser + ind] = kaiser[ind];
    }

    // Transform to frequency domain
    signal.forward(filter1D, filter1D);
}

/**
 * @param[in] subBandCenterFrequencies a vector of center frequencies for each band
 * @param[in] subBandBandwidths a vector of bandwidths for each band
 * @param[in] dt time interval of the signal
 * @param[in] fft_size length of the spectrum
 * @param[in] frequency a vector of frequencies
 * @param[in] beta parameter for the kaiser filter
 * @param[out] _filter1D one dimensional kaiser filter in frequency domain
 */
template <class T>
void
isce3::signal::Filter<T>::
constructRangeBandpassKaiser(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             int fft_size,
                             std::valarray<double>& frequency,
                             double beta,
                             std::valarray<std::complex<T>>& _filter1D)
{
    // construct a kaiser bandpass filter in frequency domian
    // which may have several bands defined by centerferquencies and
    // subBandBandwidths
    for (size_t i = 0; i<subBandCenterFrequencies.size(); ++i){

        //frequency of the lower bound of this band
        double fL = subBandCenterFrequencies[i] - subBandBandwidths[i]/2;

        //frequency of the higher bound of this band
        double fH = subBandCenterFrequencies[i] + subBandBandwidths[i]/2;

        //index of frequencies for fL and fH
        int indL;
        indexOfFrequency(dt, fft_size, fL, indL);
        int indH;
        indexOfFrequency(dt, fft_size, fH, indH);
        // std::cout << " - fL: "<< fL << " , fH: " << fH << " indL: " << indL << " , indH: " << indH << std::endl;

        // bessel_i0 of beta
        double bessel_i0_beta = isce3::math::bessel_i0(beta);

        if (fL<0 && fH>=0){
            for (size_t ind = indL; ind < fft_size; ++ind){
                double fre = fL + (ind - indL) / (dt * fft_size);
                double tmp  = 2.0 * fre * dt;
                double kaiserCoefficient = isce3::math::bessel_i0(beta * sqrt(1.0 - tmp * tmp));
                _filter1D[ind] = std::complex<T>(kaiserCoefficient/bessel_i0_beta, 0.0);
            }
            for (size_t ind = 0; ind < indH; ++ind){
                double fre = ind / (dt * fft_size);
                double tmp  = 2.0 * fre * dt;
                double kaiserCoefficient = isce3::math::bessel_i0(beta * sqrt(1.0 - tmp * tmp));
                _filter1D[ind] = std::complex<T>(kaiserCoefficient/bessel_i0_beta, 0.0);
            }

        }else{
            for (size_t ind = indL; ind < indH; ++ind){
                double fre = (ind - 0.5 * (indL + indH)) / (dt * fft_size);
                double tmp  = 2.0 * fre * dt;
                double kaiserCoefficient = isce3::math::bessel_i0(beta * sqrt(1.0 - tmp * tmp));
                _filter1D[ind] = std::complex<T>(kaiserCoefficient/bessel_i0_beta, 0.0);
            }
        }
    }
}

/**
* @param[in] refDoppler Doppler Centroids, in Hz, of the reference SLC w.r.t slant range axis
* @param[in] secDoppler Doppler Centroids, in Hz, of the secondary SLC w.r.t slant range axis
* @param[in] bandwidth  input bandwidth in azimuth of the pair of SLCs, in Hz
* @param[in] prf pulse repetition frequency, in Hz
* @param[in] windowParameter window parameter of the filter
* @param[in] signal a block of data to filter
* @param[in] spectrum of the block of data
* @param[in] ncols number of columns of the block of data
* @param[in] nrows number of rows of the block of data
* @return new bandwidth
*/
template <class T>
double
isce3::signal::Filter<T>::
constructAzimuthCommonBandFilter(const std::valarray<double> & refDoppler,
                    const std::valarray<double> & secDoppler,
                    double bandwidth,
                    double prf,
                    double windowParameter,
                    size_t ncols,
                    size_t nrows,
                    std::string& filterType)
{
    if (filterType=="cosine"){
        // Cosine filter is constructed in the frequency domain
        return constructAzimuthCommonBandCosineFilter(
                            refDoppler,
                            secDoppler,
                            bandwidth,
                            prf, windowParameter,
                            ncols, nrows);

    }
    if (filterType=="kaiser"){
        // Kaiser filter is constructed in the time domain
        return constructAzimuthCommonBandKaiserFilter(
                            refDoppler,
                            secDoppler,
                            bandwidth,
                            prf, windowParameter,
                            ncols, nrows);
    }

    std::string err_str = filterType + " filter has not been implemented";
    pyre::journal::error_t err(
        "isce.signal.Filter.constructAzimuthCommonBandFilter");
    err << err_str << pyre::journal::endl;
    throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
        err_str);
}

/**
* @param[in] refDoppler Doppler Centroids, in Hz, of the reference SLC w.r.t slant range axis
* @param[in] secDoppler Doppler Centroids, in Hz, of the secondary SLC w.r.t slant range axis
* @param[in] bandwidth  input bandwidth in azimuth of the pair of SLCs, in Hz
* @param[in] prf pulse repetition frequency, in Hz
* @param[in] beta parameter for raised cosine filter
* @param[in] signal a block of data to filter
* @param[in] spectrum of the block of data
* @param[in] ncols number of columns of the block of data
* @param[in] nrows number of rows of the block of data
*/
template <class T>
double
isce3::signal::Filter<T>::
constructAzimuthCommonBandCosineFilter(const std::valarray<double> & refDoppler,
                        const std::valarray<double> & secDoppler,
                        double bandwidth,
                        double prf,
                        double beta,
                        size_t ncols,
                        size_t nrows)
{
    _filter.resize(ncols*nrows);

    // Pedestal-dependent frequency offset for transition region
    const double df = 0.5 * bandwidth * beta;
    // Compute normalization factor for preserving average power between input
    // data and filtered data. Assumes both filter and input signal have flat
    // spectra in the passband.
    //const double norm = std::sqrt(input_BW / BW);
    const double norm = 1.0;

    // we probably need to give next power of 2 ???
    int fft_size = nrows;
    // Construct vector of frequencies
    std::valarray<double> frequency(fft_size);
    fftfreq(1.0/prf, frequency);

    // mean doppler center frequency and
    // dopper frequency shift between Reference and Secondary
    double meanDopCenterFreq = 0.0;
    double meanDopCenterFreqShift = 0.0;

    // Loop over range bins
    for (int j = 0; j < ncols; ++j) {
        double refFreq = refDoppler[j];
        double secFreq = secDoppler[j];

        double fmid =  0.5 * (refFreq + secFreq);
        double fshift = std::abs(refFreq - secFreq);

        if (fshift > bandwidth) {
            pyre::journal::error_t err(
                    "isce.signal.Filter.constructAzimuthCommonBandCosineFilter");
            err << "Bandwith is less than frequency shift"
                << pyre::journal::endl;
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Bandwith is less than frequency shift");
        }

        meanDopCenterFreqShift += fshift;
        meanDopCenterFreq += fmid;
        // Compute filter
        for (size_t i = 0; i < frequency.size(); ++i) {

            // Get the absolute value of shifted frequency
            double freq = std::abs(frequency[i] - fmid);

            // Wrap the frequency within the RPF
            freq = freq - int(freq/prf) * prf;

            // Passband
            if (freq <= (0.5 * bandwidth - df)) {
                _filter[i*ncols+j] = std::complex<T>(norm, 0.0);

            // Transition region
            } else if (freq > (0.5 * bandwidth - df) && freq <= (0.5 * bandwidth + df)) {
                _filter[i*ncols+j] = std::complex<T>(norm * 0.5 *
                                    (1.0 + std::cos(M_PI / (bandwidth*beta) *
                                    (freq - 0.5 * (1.0 - beta) * bandwidth))), 0.0);

            // Stop band
            } else {
                _filter[i*ncols+j] = std::complex<T>(0.0, 0.0);
            }
        }
    }

    // Normalize the filter
    const std::complex<T> filtNorm(fft_size, fft_size);
    for (int j = 0; j < ncols; ++j) {
        for (size_t i = 0; i < frequency.size(); ++i) {
            _filter[i*ncols+j] /= filtNorm;
        }
    }

    meanDopCenterFreq /= ncols;
    meanDopCenterFreqShift /= ncols;
    std::cout << " - mean doppler center freq:" << meanDopCenterFreq << std::endl;
    std::cout << " - mean doppler center freq shift:" << meanDopCenterFreqShift << std::endl;

    return (bandwidth - meanDopCenterFreqShift);
}

/**
* @param[in] refDoppler Doppler Centroids, in Hz, of the reference SLC w.r.t slant range axis
* @param[in] secDoppler Doppler Centroids, in Hz, of the secondary SLC w.r.t slant range axis
* @param[in] bandwidth  input bandwidth in azimuth of the pair of SLCs, in Hz
* @param[in] prf pulse repetition frequency, in Hz
* @param[in] beta parameter for kaiser filter
* @param[in] signal a block of data to filter
* @param[in] spectrum of the block of data
* @param[in] ncols number of columns of the block of data
* @param[in] nrows number of rows of the block of data
* @returns common azimuth bandwidth
*/
template <class T>
double
isce3::signal::Filter<T>::
constructAzimuthCommonBandKaiserFilter(const std::valarray<double> & refDoppler,
                        const std::valarray<double> & secDoppler,
                        double bandwidth,
                        double prf,
                        double beta,
                        size_t ncols,
                        size_t nrows)
{
    pyre::journal::debug_t debug("isce.signal.Filter.constructAzimuthCommonBandKaiserFilter");

    _filter.resize(ncols*nrows);
    _filter = std::complex<T>(0.0, 0.0);
    int fft_size = nrows;

    std::valarray<std::complex<T>> filter1D(fft_size);
    isce3::signal::Signal<T> doppSignal;
    doppSignal.forwardRangeFFT(filter1D, filter1D, fft_size, 1);

    // azimuth mean doppler centroid frequency, in Hz
    double meanDopCenterFreq = 0.0;

    // azimuth mean doppler center frequency shift, in Hz
    double meanDopCenterFreqShift = 0.0;

    // Loop over range bins
    for (int j = 0; j < ncols; ++j) {
        // Compute the mean doppler center frequency
        double refFreq = refDoppler[j];
        double secFreq = secDoppler[j];

        double fmid =  0.5 * (refFreq + secFreq);
        double fshift = std::abs(refFreq - secFreq);

        if (bandwidth < fshift) {
            pyre::journal::error_t err(
                    "isce.signal.Filter.constructAzimuthCommonBandKaiserFilter");
            err << "Bandwith is less than frequency shift"
                << pyre::journal::endl;
            throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Bandwith is less than frequency shift");
        }

        std::valarray<std::complex<T>> kaiser;
        _design_shaped_lowpass_filter(bandwidth - fshift, prf, beta, kaiser);

        // Turn into the bandpass filter
         _lowpass2bandpass(kaiser, fmid/prf, kaiser);

        const int sizeOfKaiser = kaiser.size();
        const int halfSizeOfKaiser = (sizeOfKaiser - 1)/2;
        if (fft_size < sizeOfKaiser) {
            pyre::journal::error_t err(
                    "isce.signal.Filter.constructAzimuthCommonBandKaiserFilter");
            err << "FFT size " << fft_size <<" is less than the filter kernel size "
                << sizeOfKaiser
                << pyre::journal::endl;
            throw isce3::except::LengthError(ISCE_SRCINFO(),
                "FFT size is less than the filter kernel size");
        }

        // Zero padding the filter in the middle
        filter1D = std::complex<T>(0.0, 0.0);
        for (size_t ind = halfSizeOfKaiser; ind < sizeOfKaiser; ind ++) {
            filter1D[ind - halfSizeOfKaiser] = kaiser[ind];
        }
        for (size_t ind = 0; ind < halfSizeOfKaiser; ind ++) {
            filter1D[fft_size - halfSizeOfKaiser + ind] = kaiser[ind];
        }

        // Transform to frequency domain
        doppSignal.forward(filter1D, filter1D);

        meanDopCenterFreqShift += fshift;
        meanDopCenterFreq += fmid;

        // Copy the the filter centered at fmid to a block filter
        // and normalize the transform to recovery the input signal via
        // dividing by fft_size
        for (size_t i = 0; i < filter1D.size(); ++i) {
            _filter[i*ncols+j] = filter1D[i] / static_cast<T>(fft_size);
        }
    }

    meanDopCenterFreq /= ncols;
    meanDopCenterFreqShift /= ncols;

    debug << " - mean doppler center freq (Hz):" << meanDopCenterFreq << pyre::journal::endl;
    debug << " - mean doppler center freq shift (Hz):" << meanDopCenterFreqShift << pyre::journal::endl;

    return (bandwidth - meanDopCenterFreqShift);
}


/**
* @param[in] signal a block of data to filter.
* @param[in] spectrum of the block of the data
*/
template <class T>
void
isce3::signal::Filter<T>::
filter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum)
{
    _signal.forward(signal, spectrum);
    spectrum = spectrum*_filter;
    _signal.inverse(spectrum, signal);
}

/**
 * @param[in] N length of the signal
 * @param[in] dt sampling interval of the signal
 * @param[out] freq output vector of the frequencies
 */
void
isce3::signal::
fftfreq(double dt, std::valarray<double> &freq){

    int N = freq.size();

    // Scale factor
    const double scale = 1.0 / (N * dt);
    // Allocate vector
    // Fill in the positive frequencies
    int N_mid = (N - 1) / 2 + 1;
    for (int i = 0; i < N_mid; ++i) {
        freq[i] = scale * i;
    }
    // Fill in the negative frequencies
    int ind = N_mid;
    for (int i = -N/2; i < 0; ++i) {
        freq[ind] = scale * i;
        ++ind;
    }
}

/**
 * @param[in] dt sampling interval of the signal
 * @param[in] N length of the signal
 * @param[in] f frequency of interest
 * @param[out] n index of the frequency
 */
template <class T>
void
isce3::signal::Filter<T>::
indexOfFrequency(double dt, int N, double f, int &n)
// deterrmine the index (n) of a given frequency f
// dt: sampling rate,
// N: length of a signal
// f: frequency of interest
// Assumption: for indices 0 to (N-1)/2, frequency is positive
//              and for indices larger than (N-1)/2 frequency is negative
{
    double df = 1/(dt*N);

    if (f < 0)
        n = round(f/df + N);
    else
        n = round(f/df);
}

template <class T>
void
isce3::signal::Filter<T>::
writeFilter(size_t ncols, size_t nrows)
{
    isce3::io::Raster filterRaster("filter.bin", ncols, nrows, 1, GDT_CFloat32, "ENVI");
    filterRaster.setBlock(_filter, 0, 0, ncols, nrows);

}

template <class T>
std::tuple<int, double>
isce3::signal::Filter<T>::_kaiserord(const double ripple, const double transition_width)
{
    const double A = std::abs(ripple);  // in case somebody is confused as to what's meant
    if (A < 8) {
        // Formula for N is not valid in this range.
        pyre::journal::error_t err(
                "isce.signal.Filter._kaiserord");
        err << "Requested maximum ripple attentuation " << A
            << " is too small for the Kaiser formula."
            << pyre::journal::endl;
         throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Requested maximum ripple attentuation is too small for the Kaiser formula.");
    }

    // Kaiser's formula (as given in Oppenheim and Schafer) is for the filter
    // order, so we have to add 1 to get the number of taps.
    const double numtaps = (A - 7.95) / 2.285 / (M_PI * transition_width) + 1;

    return std::make_tuple(int(std::ceil(numtaps)), _kaiser_beta(A));
}

template <class T>
double
isce3::signal::Filter<T>::_kaiser_beta(const double ripple)
{
    if (ripple > 50) {
        return 0.1102 * (ripple - 8.7);
    }
    else if (ripple > 21){
        return 0.5842 * std::pow(ripple - 21.0, 0.4) + 0.07886 * (ripple - 21);
    }
    else {
        return 0.0;
    }
}

template <class T>
void
isce3::signal::Filter<T>::_kaiser_design(const double stopatt,
                            const double transition_width,
                            const bool force_odd_len,
                            int &n,
                            double &beta,
                            std::valarray<double> &t)
{
    std::tie(n, beta) = _kaiserord(stopatt, transition_width);
    if (force_odd_len && (n % 2 == 0)) n += 1;

    if (t.size() <= 0) t.resize(n);
    for (size_t i = 0; i < t.size(); i++) t[i] = i - (n - 1) / 2.0;
}

template <class T>
void
isce3::signal::Filter<T>::_kaiser_irf(const std::valarray<double> &t,
                         const double beta,
                         std::valarray<std::complex<T>> &irf)
{
    const double alpha =  beta / M_PI;
    const double beta0 = isce3::math::bessel_i0(beta);

    if (irf.size() <= 0) irf.resize(t.size());
    for (size_t i = 0; i < t.size(); i++) {
        auto val = std::complex<T>(t[i] * t[i] - alpha * alpha, 0.0);
        auto sincVal = isce3::math::sinc<T>(std::sqrt(val)) / beta0;
        irf[i] = std::complex<T>(sincVal.real(), 0.0);
    }
}

template <class T>
void
isce3::signal::Filter<T>::_kaiser(const int n,
                     const double beta,
                     std::valarray<std::complex<T>> &coeffs)
{
    const double beta0 = isce3::math::bessel_i0(beta);

    if (coeffs.size() <= 0) coeffs.resize(n);

    if (n == 1) {
        coeffs[0] = std::complex<T>(1.0,0.0);
        return;
    }
    const double alpha = (n - 1) / 2.0;
    for (size_t i = 0; i < n; i++) {
        const double t = (i - alpha)/alpha;
        coeffs[i] = isce3::math::bessel_i0(beta * std::sqrt(1.0 - t * t)) / beta0;
    }
}

template <class T>
void
isce3::signal::Filter<T>::_lowpass2bandpass(const std::valarray<std::complex<T>> &low_pass_filter,
                    const double fc,
                    std::valarray<std::complex<T>> &band_pass_filter)
{
    if (band_pass_filter.size() <= 0) band_pass_filter.resize(low_pass_filter.size());
    const size_t n = low_pass_filter.size();
    for (size_t i = 0; i < n; i++) {
        const double t = (i - (n - 1) / 2.0);
        const double phase = 2.0 * M_PI * fc * t;
        const std::complex<T> ramp_phase(std::cos(phase), std::sin(phase));
        band_pass_filter[i] = low_pass_filter[i] * ramp_phase;
    }
}

template <class T>
void
isce3::signal::Filter<T>::_design_shaped_lowpass_filter(const double bandwidth,
                                           const double fs,
                                           const double window_shape,
                                           std::valarray<std::complex<T>> &coeffs,
                                           const double stopatt,
                                           const double transition_width,
                                           const bool force_odd_len)
{
    // Normalized bandwdith
    const double bw = bandwidth / fs;

    // Transition width is specified in terms of output bandwidth, so scale to
    // get width at sample rate of filter.
    const double tw = transition_width * bw;

    if ((bw + tw / 2.0) > 1.0) {
        pyre::journal::error_t err(
                "isce.signal.Filter._design_shaped_lowpass_filter");
        err << "Passband + transition cannot exceed Nyquist"
            << pyre::journal::endl;
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Passband + transition cannot exceed Nyquist");
    }

    int n = 0;
    double beta = 0.0;
    std::valarray<double> t;
    std::valarray<std::complex<T>> irf;
    std::valarray<std::complex<T>> kaiser;

    _kaiser_design(stopatt, tw, force_odd_len, n, beta, t);
    _kaiser_irf(t * bw, window_shape, irf);
    _kaiser(n, beta, kaiser);

    if (coeffs.size() <= 0) coeffs.resize(n);

    for (size_t i = 0; i < n; i++) {
        coeffs[i] = irf[i] * kaiser[i] * bw;
    }
}

template class isce3::signal::Filter<float>;
template class isce3::signal::Filter<double>;

