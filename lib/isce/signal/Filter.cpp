// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#include "Filter.h"

/**
 * @param[in] rangeSamplingFrequency range sampling frequency
 * @param[in] subBandCenterFrequencies a vector of center frequencies for each band
 * @param[in] subBandBandwidths a vector of bandwidths for each band
 * @param[in] signal a block of data to filter
 * @param[in] spectrum a block of spectrum, which is internally used for FFT computations
 * @param[in] ncols number of columns of the block of data
 * @param[in] nrows number of rows of the block of data
 * @param[in] filterType type of the band-pass filter
 */
template <class T>
void
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

    int nfft = ncols;

    _filter.resize(nfft*nrows);
    std::valarray<std::complex<T>> _filter1D(nfft); //
    _filter1D = std::complex<T>(0.0,0.0);

    std::valarray<double> frequency(nfft);
    double dt = 1.0/rangeSamplingFrequency;
    fftfreq(nfft, dt, frequency);

    if (filterType=="boxcar"){
        constructRangeBandpassBoxcar(
                            subBandCenterFrequencies,
                            subBandBandwidths,
			    dt,
                            nfft,
	                    _filter1D);
        
    } else if (filterType=="cosine"){
        double beta = 0.25;
        constructRangeBandpassCosine(subBandCenterFrequencies,
                            subBandBandwidths,
                            dt,
                            frequency,
                            beta,
                            _filter1D); 

    } else {
        std::cout << filterType << " filter has not been implemented" << std::endl;
    }
    
    //construct a block of the filter
    for (size_t line = 0; line < nrows; line++ ){
        for (size_t col = 0; col < nfft; col++ ){
            _filter[line*nfft+col] = _filter1D[col];
        }
    }

    _signal.forwardRangeFFT(signal, spectrum, ncols, nrows);
    _signal.inverseRangeFFT(spectrum, signal, ncols, nrows);
   
}

/**
 * @param[in] subBandCenterFrequencies a vector of center frequencies for each band
 * @param[in] subBandBandwidths a vector of bandwidths for each band
 * @param[in] dt samplig rate of the signal
 * @param[in] nfft length of the spectrum
 * @param[out] _filter1D one dimensional boxcar bandpass filter in frequency domain 
 */
template <class T>
void
isce::signal::Filter<T>::
constructRangeBandpassBoxcar(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
                             int nfft,
                             std::valarray<std::complex<T>>& _filter1D)
{
    // construct a boxcar bandpass filter in frequency domian 
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
        indexOfFrequency(dt, nfft, fL, indL); 
        int indH;
        indexOfFrequency(dt, nfft, fH, indH);
        std::cout << "fL: "<< fL << " , fH: " << fH << " indL: " << indL << " , indH: " << indH << std::endl;
        if (fL<0 && fH>=0){
            for (size_t ind = indL; ind < nfft; ++ind){
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
 * @param[in] dt samplig rate of the signal
 * @param[in] frequency a vector of frequencies
 * @param[in] beta parameter for the raised cosine filter (0 <= beta <= 1)
 * @param[out] _filter1D one dimensional boxcar bandpass filter in frequency domain
 */
template <class T>
void
isce::signal::Filter<T>::
constructRangeBandpassCosine(std::valarray<double> subBandCenterFrequencies,
                             std::valarray<double> subBandBandwidths,
                             double dt,
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
* @param[in] refDoppler Doppler polynomial of the reference SLC
* @param[in] secDoppler Doppler polynomial of the secondary SLC
* @param[in] bandwidth common bandwidth in azimuth
* @param[in] prf pulse repetition frequency
* @param[in] beta parameter for raised cosine filter
* @param[in] signal a block of data to filter
* @param[in] spectrum of the block of data
* @param[in] ncols number of columns of the block of data
* @param[in] nrows number of rows of the block of data
*/
template <class T>
void
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
                _filter[i*ncols+j] = std::complex<T>(norm, 0.0);

            // Transition region
            } else if (freq > (0.5 * bandwidth - df) && freq <= (0.5 * bandwidth + df)) {
                _filter[i*ncols+j] = std::complex<T>(norm * 0.5 * 
                                    (1.0 + std::cos(M_PI / (bandwidth*beta) * 
                                    (freq - 0.5 * (1.0 - beta) * bandwidth))), 0.0);

            // Stop band
            } else {
                _filter[i+j*ncols] = std::complex<T>(0.0, 0.0);
            }
        }
    }

    _signal.forwardAzimuthFFT(signal, spectrum, ncols, nrows);
    _signal.inverseAzimuthFFT(spectrum, signal, ncols, nrows);
}

/**
* @param[in] signal a block of data to filter.
* @param[in] spectrum of the block of the data 
*/
template <class T>
void
isce::signal::Filter<T>::
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
template <class T>
void
isce::signal::Filter<T>::
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
 * @param[in] dt sampling interval of the signal
 * @param[in] N length of the signal
 * @param[in] f frequency of interest
 * @param[out] n index of the frequency
 */
template <class T>
void
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
}

template <class T>
void
isce::signal::Filter<T>::
writeFilter(size_t ncols, size_t nrows)
{
    isce::io::Raster filterRaster("filter.bin", ncols, nrows, 1, GDT_CFloat32, "ENVI");
    filterRaster.setBlock(_filter, 0, 0, ncols, nrows);

}

template class isce::signal::Filter<float>;
template class isce::signal::Filter<double>;

