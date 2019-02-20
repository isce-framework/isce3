#include "shiftSignal.h"

template<typename T>
void isce::signal::
shiftSignal(std::valarray<T> & data,
            std::valarray<T> & dataShifted,
            std::valarray<std::complex<T>> & spectrum,
            size_t ncols, size_t nrows,
            const double shift, isce::signal::Signal<T> & sigObj)
{

    // buffer for the frequency domain phase ramp introduced by 
    // the constant shift in time domain
    std::valarray<std::complex<T>> phaseRamp(ncols * nrows);

    // compute the frequency domain phase ramp
    frequencyResponse(ncols, nrows, shift, phaseRamp);

    
    shiftSignal(data, dataShifted, 
                spectrum, phaseRamp,
                sigObj);

    dataShifted /= ncols;
}

template<typename T>
void isce::signal::
shiftSignal(std::valarray<std::complex<T>> & data,
                std::valarray<std::complex<T>> & dataShifted,
                std::valarray<std::complex<T>> & spectrum,
                size_t ncols, size_t nrows,
                const double shift, isce::signal::Signal<T> & sigObj)
{

    // buffer for the frequency domain phase ramp introduced by
    // the constant shift in time domain
    std::valarray<std::complex<T>> phaseRamp(ncols * nrows);

    // compute the frequency domain phase ramp
    frequencyResponse(ncols, nrows, shift, phaseRamp);


    shiftSignal(data, dataShifted,
                spectrum, phaseRamp,
                sigObj);

    dataShifted /= ncols;

}

template<typename T>
void isce::signal::
shiftSignal(std::valarray<T> & data,
            std::valarray<T> & dataShifted,
            std::valarray<std::complex<T>> & spectrum,
            std::valarray<std::complex<T>> & phaseRamp,
            isce::signal::Signal<T> & sigObj) {

    sigObj.forward(data, spectrum);

    spectrum *= phaseRamp;

    sigObj.inverse(spectrum, dataShifted);

}

template<typename T>
void isce::signal::
shiftSignal(std::valarray<std::complex<T>> & data,
                std::valarray<std::complex<T>> & dataShifted,
                std::valarray<std::complex<T>> & spectrum,
                std::valarray<std::complex<T>> & phaseRamp,
                isce::signal::Signal<T> & sigObj) {

    sigObj.forward(data, spectrum);

    spectrum *= phaseRamp;

    sigObj.inverse(spectrum, dataShifted);

}

template<typename T>
void isce::signal::
frequencyResponse(size_t nfft, size_t blockRows, const double shift, 
        std::valarray<std::complex<T>> & shiftImpact)
{
    // range frequencies given nfft and oversampling factor
    std::valarray<double> rangeFrequencies(nfft);

    // sampling interval in range
    double dt = 1.0;

    // get the vector of range frequencies
    //filter object
    isce::signal::Filter<T> tempFilter;
    tempFilter.fftfreq(nfft, dt, rangeFrequencies);

    // in the process of upsampling the SLCs, creating upsampled interferogram
    // and then looking down the upsampled interferogram to the original size of
    // the SLCs, a shift is introduced in range direction.
    // As an example for a signal with length of 5 and :
    // original sample locations:   0       1       2       3        4
    // upsampled sample locations:  0   0.5 1  1.5  2  2.5  3   3.5  4   4.5
    // Looked dow sample locations:   0.25    1.25    2.25    3.25    4.25
    // Obviously the signal after looking down would be shifted by 0.25 pixel in
    // range comared to the original signal. Since a shift in time domain introduces
    // a liner phase in frequency domain, we compute the impact in frequency domain.

    // the constant shift based on the oversampling factor

    // compute the frequency response of the subpixel shift in range direction
    std::valarray<std::complex<T>> shiftImpactLine(nfft);
    for (size_t col=0; col<shiftImpactLine.size(); ++col){
        double phase = -1.0*shift*2.0*M_PI*rangeFrequencies[col];
        shiftImpactLine[col] = std::complex<T> (std::cos(phase),
                                                    std::sin(phase));
    }

    // The imapct is the same for each range line. Therefore copying the line for the block
    for (size_t line = 0; line < blockRows; ++line) {
        shiftImpact[std::slice(line*nfft, nfft, 1)] = shiftImpactLine;
    }
}


template void isce::signal::
shiftSignal(std::valarray<float> & data,
            std::valarray<float> & dataShifted,
            std::valarray<std::complex<float>> & spectrum,
            size_t ncols, size_t nrows,
            const double shift, isce::signal::Signal<float> & sigObj);


template void isce::signal::
shiftSignal(std::valarray<double> & data,
            std::valarray<double> & dataShifted,
            std::valarray<std::complex<double>> & spectrum,
            size_t ncols, size_t nrows,
            const double shift, isce::signal::Signal<double> & sigObj);

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
            frequencyResponse(size_t nfft, size_t blockRows, 
            const double shift,
            std::valarray<std::complex<float>> & shiftImpact);

template void isce::signal::
            frequencyResponse(size_t nfft, size_t blockRows,
            const double shift,
            std::valarray<std::complex<double>> & shiftImpact);

