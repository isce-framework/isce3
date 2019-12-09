// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-

#include "shiftSignal.h"

/**
 * @param[in] data input data to be shifted
 * @param[out] dataShifted output data after the shift
 * @param[in] spectrum a memory block for the spectrum of the data
 * @param[in] ncols number of columns
 * @param[in] nrows number of rows
 * @param[in] shiftX constant shift in X direction (columns)
 * @param[in] shiftY constant shift in range Y direction (rows)
 * @param[in] sigObj signal object
 */
template<typename T, typename U>
void isce::signal::
shiftSignal(std::valarray<T> & data,
            std::valarray<T> & dataShifted,
            std::valarray<std::complex<U>> & spectrum,
            size_t ncols, size_t nrows,
            const double shiftX, const double shiftY,
            isce::signal::Signal<U> & sigObj)
{

    // variable for the total length of the signal used in FFT computation
    size_t fft_size = 1;
    std::valarray<std::complex<U>> phaseRamp(ncols*nrows);
    if (not isce::core::compareFloatingPoint(shiftX, 0.0)) {
        // buffer for the frequency domain phase ramp introduced by
        // the constant shift in time domain

        // compute the frequency domain phase ramp
        frequencyResponseRange(ncols, nrows, shiftX, phaseRamp);

        // so far the fft length is ncols
        fft_size = ncols;
    }

    if (not isce::core::compareFloatingPoint(shiftY, 0.0)) {

        // buffer for the frequency domain phase ramp introduced by
        // the constant shift in time domain
        std::valarray<std::complex<U>> phaseRampY(ncols * nrows);

        // compute the frequency domain phase ramp
        frequencyResponseAzimuth(ncols, nrows, shiftY, phaseRampY);

        // taking into account nrows for the fft length 
        fft_size *=nrows;

        if (not isce::core::compareFloatingPoint(shiftX, 0.0))
            // if there was a Shift in X, the impact in frequency domain
            // is multiplied by the impact in Y direction
            // F(X,Y) <--> f(x,y)
            // exp(1J(x0+y0))*F(X,Y) <--> f(x-x0,y-y0)
            phaseRamp *= phaseRampY;

        else
            phaseRamp = phaseRampY;

    }

    if (not isce::core::compareFloatingPoint(shiftX, 0.0) || 
            not isce::core::compareFloatingPoint(shiftY, 0.0)) {

        // if any shift requested at all
        shiftSignal(data, dataShifted,
                spectrum, phaseRamp,
                sigObj);
         
        // since FFTW is not normalized, here we divide by total length of fft
        dataShifted /= fft_size;

    } else {
        // if no shift requested, return the original signal 
        dataShifted = data;
    }

}


/**
 * @param[in] data input data to be shifted
 * @param[out] dataShifted output data after the shift
 * @param[in] spectrum a memory block for the spectrum of the data
 * @param[in] phaseRamp frequency domain response of the constant shift
 * @param[in] sigObj signal object
 */
template<typename T, typename U>
void isce::signal::
shiftSignal(std::valarray<T> & data,
            std::valarray<T> & dataShifted,
            std::valarray<std::complex<U>> & spectrum,
            std::valarray<std::complex<U>> & phaseRamp,
            isce::signal::Signal<U> & sigObj) {

    // forward FFT of the data
    sigObj.forward(data, spectrum);

    // mutiply the spectrum by the impact of the shift in frequency domain
    spectrum *= phaseRamp;

    // inverse fft to get back the shifted signal
    sigObj.inverse(spectrum, dataShifted);

}

/**
 * @param[in] ncols number of columns
 * @param[in] nrows number of rows
 * @param[in] shift constant shift in X direction (columns)
 * @param[out] shiftImpact impact of the shift in frequency domain
 */
template<typename T>
void isce::signal::
frequencyResponseRange(size_t ncols, size_t nrows, const double shift,
        std::valarray<std::complex<T>> & shiftImpact)
{
    // range frequencies given fft_size and oversampling factor
    std::valarray<double> rangeFrequencies(ncols);

    // sampling interval in range
    double dt = 1.0;

    // get the vector of range frequencies
    fftfreq(dt, rangeFrequencies);

    // compute the impact of the shift in the frequency domain
    std::valarray<std::complex<T>> shiftImpactLine(ncols);
    for (size_t col=0; col<shiftImpactLine.size(); ++col){
        double phase = -1.0*shift*2.0*M_PI*rangeFrequencies[col];
        shiftImpactLine[col] = std::complex<T> (std::cos(phase),
                                                    std::sin(phase));
    }

    // The imapct is the same for each range line. Therefore copying the line for the block
    for (size_t line = 0; line < nrows; ++line) {
        shiftImpact[std::slice(line*ncols, ncols, 1)] = shiftImpactLine;
    }
}

/**
 * @param[in] ncols number of columns
 * @param[in] nrows number of rows
 * @param[in] shift constant shift in Y direction (columns)
 * @param[out] shiftImpact impact of the shift in frequency domain
 */
template<typename T>
void isce::signal::
frequencyResponseAzimuth(size_t ncols, size_t nrows, const double shift,
        std::valarray<std::complex<T>> & shiftImpact)
{
    // azimuth frequencies given fft_size and oversampling factor
    std::valarray<double> frequencies(nrows);

    // sampling interval in time
    double dt = 1.0;

    // get the vector of range frequencies
    fftfreq(dt, frequencies);

    // compute the impact of the azimuth shift in frequency domain
    std::valarray<std::complex<T>> shiftImpactLine(nrows);
    for (size_t row = 0; row < shiftImpactLine.size(); ++row){
        double phase = -1.0*shift*2.0*M_PI*frequencies[row];
        shiftImpactLine[row] = std::complex<T> (std::cos(phase),
                                                    std::sin(phase));
    }

    // The imapct is the same for each column. Therefore copying for all columns of the block
    for (size_t col = 0; col < ncols; ++col) {
        shiftImpact[std::slice(col, nrows, ncols)] = shiftImpactLine;
    }
}


template void isce::signal::
shiftSignal(std::valarray<float> & data,
            std::valarray<float> & dataShifted,
            std::valarray<std::complex<float>> & spectrum,
            size_t ncols, size_t nrows,
            const double shiftX, const double shiftY,
            isce::signal::Signal<float> & sigObj);

template void isce::signal::
shiftSignal(std::valarray<double> & data,
            std::valarray<double> & dataShifted,
            std::valarray<std::complex<double>> & spectrum,
            size_t ncols, size_t nrows,
            const double shiftX, const double shiftY,
            isce::signal::Signal<double> & sigObj);

template void isce::signal::
shiftSignal(std::valarray<std::complex<float>> & data,
            std::valarray<std::complex<float>> & dataShifted,
            std::valarray<std::complex<float>> & spectrum,
            size_t ncols, size_t nrows,
            const double shiftX, const double shiftY,
            isce::signal::Signal<float> & sigObj);

template void isce::signal::
shiftSignal(std::valarray<std::complex<double>> & data,
            std::valarray<std::complex<double>> & dataShifted,
            std::valarray<std::complex<double>> & spectrum,
            size_t ncols, size_t nrows,
            const double shiftX, const double shiftY,
            isce::signal::Signal<double> & sigObj);

template void isce::signal::
shiftSignal(std::valarray<float> & data,
            std::valarray<float> & dataShifted,
            std::valarray<std::complex<float>> & spectrum,
            std::valarray<std::complex<float>> & phaseRamp,
            isce::signal::Signal<float> & sigObj);

template void isce::signal::
shiftSignal(std::valarray<double> & data,
            std::valarray<double> & dataShifted,
            std::valarray<std::complex<double>> & spectrum,
            std::valarray<std::complex<double>> & phaseRamp,
            isce::signal::Signal<double> & sigObj);

template void isce::signal::
shiftSignal(std::valarray<std::complex<float>> & data,
            std::valarray<std::complex<float>> & dataShifted,
            std::valarray<std::complex<float>> & spectrum,
            std::valarray<std::complex<float>> & phaseRamp,
            isce::signal::Signal<float> & sigObj);

template void isce::signal::
shiftSignal(std::valarray<std::complex<double>> & data,
            std::valarray<std::complex<double>> & dataShifted,
            std::valarray<std::complex<double>> & spectrum,
            std::valarray<std::complex<double>> & phaseRamp,
            isce::signal::Signal<double> & sigObj);

template void isce::signal::
            frequencyResponseRange(size_t fft_size, size_t blockRows,
            const double shift,
            std::valarray<std::complex<float>> & shiftImpact);

template void isce::signal::
            frequencyResponseRange(size_t fft_size, size_t blockRows,
            const double shift,
            std::valarray<std::complex<double>> & shiftImpact);

template void isce::signal::
            frequencyResponseAzimuth(size_t fft_size, size_t blockRows,
            const double shift,
            std::valarray<std::complex<float>> & shiftImpact);

template void isce::signal::
            frequencyResponseAzimuth(size_t fft_size, size_t blockRows,
            const double shift,
            std::valarray<std::complex<double>> & shiftImpact);

