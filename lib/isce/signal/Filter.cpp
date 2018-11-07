// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#include "Filter.h"

/**
 * @param[in] range sampling frequency
 * @param[in] a vector of center frequencies for each band
 * @param[in] a vector of bandwidths for each band
 * @param[in] a block of data to filter
 * @param[in] a block of spectrum, which is internally used for FFT computations
 * @param[in] number of columns of the block of data
 * @param[in] number of rows of the block of data
 * @param[in] type of the band-pass filter
 */
template <class T>
T
isce::signal::Filter<T>::
constructRangeBandpassFilter(double rangeSamplingFrequency,
                                std::valarray<double> subBandCenterFrequencies,
                                std::valarray<double> subBandBandwidths,
                                std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                size_t ncols,
                                size_t nrows,
                                std::string filterType)
{
    if (filterType=="boxcar"){
        constructRangeBandpassBoxcar(rangeSamplingFrequency,
                             subBandCenterFrequencies,
                             subBandBandwidths,
                             ncols,
                             nrows);
        
    }    

    _signal.forwardRangeFFT(signal, spectrum, ncols, nrows, ncols, nrows);
    _signal.inverseRangeFFT(spectrum, signal, ncols, nrows, ncols, nrows);
   
}

/**
 * @param[in] range sampling frequency
 * @param[in] a vector of center frequencies for each band
 * @param[in] a vector of bandwidths for each band
 * @param[in] number of columns of the block of data
 * @param[in] number of rows of the block of data
 */
template <class T>
T
isce::signal::Filter<T>::
constructRangeBandpassBoxcar(double rangeSamplingFrequency,
                             std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             size_t ncols,
                             size_t nrows)
{
    // construct a boxcar bandpass filter in frequency domian 
    // which may have several bands defined by centerferquencies and 
    // subBandBandwidths

    int nfft = ncols;

    _filter.resize(nfft*nrows);
    std::valarray<std::complex<T>> _filter1col(nfft); // 
    _filter1col = std::complex<T>(0.0,0.0);

    std::valarray<double> frequency(nfft);
    double dt = 1.0/rangeSamplingFrequency;
    fftfreq(nfft, dt, frequency);

    for (size_t i = 0; i<subBandCenterFrequencies.size(); ++i){
        //frequency of the lower bound of this band
        double fL = subBandCenterFrequencies[i] - subBandBandwidths[i]/2;

        //frequency of the higher bound of this band
        double fH = subBandCenterFrequencies[i] + subBandBandwidths[i]/2;

        //index of frequencies for fL and fH
        int indL; 
        indexOfFrequency(dt, nfft, fL, indL); 
        int indH;
        indexOfFrequency(dt, nfft, fH, indH);

        for (size_t ind = indL; ind < indH; ++ind){
            _filter1col[ind] = std::complex<T>(1.0, 1.0);
        }

    }

    for (size_t line = 0; line < nrows; line++ ){
        for (size_t col = 0; col < nfft; col++ ){
            _filter[line*nfft+col] = _filter1col[col];
        }
    }

}

/**
* @param[in] Doppler polynomial of the reference SLC
* @param[in] Doppler polynomial of the secondary SLC
* @param[in] common bandwidth in azimuth
* @param[in] pulse repetition frequency
* @param[in] beta parameter
* @param[in] a block of data to filter
* @param[in] spectrum of the block of data
* @param[in] number of columns of the block of data
* @param[in] number of rows of the block of data
*/
template <class T>
T
isce::signal::Filter<T>::
constructAzimuthCommonbandFilter(const isce::core::Poly2d & refDoppler,
                        const isce::core::Poly2d & secDoppler,
                        double bandwidth,
                        double prf,
                        double beta,
                        std::valarray<std::complex<T>> &signal,
                        std::valarray<std::complex<T>> &spectrum,
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
    int nfft = nrows;
    // Construct vector of frequencies
    std::valarray<double> frequency(nfft);
    fftfreq(nfft, 1.0/prf, frequency);
    
    // Loop over range bins
    for (int j = 0; j < ncols; ++j) {

        // Compute center frequency of common band
        const double fmid = 0.5 * (refDoppler.eval(0, j) + secDoppler.eval(0, j));

        // Compute filter
        for (size_t i = 0; i < frequency.size(); ++i) {

            // Get the absolute value of shifted frequency
            const double freq = std::abs(frequency[i] - fmid);

            // Passband
            if (freq <= (0.5 * bandwidth - df)) {
                _filter[i+j*ncols] = std::complex<T>(norm, 0.0);

            // Transition region
            } else if (freq > (0.5 * bandwidth - df) && freq <= (0.5 * bandwidth + df)) {
                _filter[i+j*ncols] = std::complex<T>(norm * 0.5 * 
                                    (1.0 + std::cos(M_PI / (bandwidth*beta) * 
                                    (freq - 0.5 * (1.0 - beta) * bandwidth))), 0.0);

            // Stop band
            } else {
                _filter[i+j*ncols] = std::complex<T>(0.0, 0.0);
            }
        }
    }
    
    _signal.forwardAzimuthFFT(signal, spectrum, ncols, nrows, ncols, nrows);
    _signal.inverseAzimuthFFT(spectrum, signal, ncols, nrows, ncols, nrows);
}

/**
* @param[in] a block of data to filter.
* @param[in] spectrum of the block of the data 
*/
template <class T>
T
isce::signal::Filter<T>::
filter(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum)
{
    _signal.forward(signal, spectrum);
    spectrum = spectrum*_filter;
    _signal.inverse(spectrum, signal);   
}

/**
 * @param[in] length of the signal
 * @param[in] sampling interval of the signal
 * @param[out] output vector of the frequencies 
 */
template <class T>
T
isce::signal::Filter<T>::
//void isce::signal::Filter::
fftfreq(int N, double dt, std::valarray<double> &freq){

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
 * @param[in] sampling interval of the signal
 * @param[in] length of the signal
 * @param[in] frequency of interest
 * @param[out] index of the frequency
 */
template <class T>
T
isce::signal::Filter<T>::
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
    return n;
}

template class isce::signal::Filter<float>;
template class isce::signal::Filter<double>;

