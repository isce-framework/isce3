
#include "Signal.h"
#include <iostream>

isce::signal::Signal::
Signal(std::vector< std::complex<float>>& data, int oversample) {
    _nfft = data.size();
    _oversample = oversample;
    _data = data;
    _spectrum.resize(_nfft);
    _filteredData.resize(_oversample*_nfft);
    _filteredSpectrum.resize(_oversample*_nfft);
}

isce::signal::Signal::
~Signal() {
    fftwf_destroy_plan(_plan_fwd);
    fftwf_destroy_plan(_plan_inv);
}

void isce::signal::Signal::
initFFTPlans() {

    _plan_fwd = fftwf_plan_dft_1d(
        _nfft,
        (fftwf_complex *) (&_data[0]),
        (fftwf_complex *) (&_spectrum[0]),
        FFTW_FORWARD,
        FFTW_ESTIMATE);

    _plan_inv = fftwf_plan_dft_1d(
        _oversample*_nfft,
        (fftwf_complex *) (&_filteredSpectrum[0]),
        (fftwf_complex *) (&_filteredData[0]),
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

}

void isce::signal::Signal::
bandPass(double samplingRate, double subBandCenterFrequency, double subBandBandwidth)
{
    std::vector<double> Band;
    Band.push_back(subBandCenterFrequency - subBandBandwidth/2.0);
    Band.push_back(subBandCenterFrequency + subBandBandwidth/2.0);
    bandPass(samplingRate, Band);
    
}

void isce::signal::Signal::
bandPass(double samplingRate, std::vector<double> Band)
{
    // index of the lower bound of the sub-band
    int indLowerBound = _indexOfFrequency(samplingRate, _nfft, Band[0]);
    // index of the higher bound of the sub-band
    int indHigherBound = _indexOfFrequency(samplingRate, _nfft, Band[1]);

    std::vector<float> weights(_nfft);
    for (int i=indLowerBound; i<indHigherBound; i++) // Or i<indHigherBound + 1
        weights[i] = 1.0;

    Filter(weights);
}

int isce::signal::Signal::
_indexOfFrequency(double S, int N, double f)
{
// deterrmine the index (n) of a given frequency f for a signal 
// with sampling rate of S and length of N
// Assumption: for indices 0 to (N-1)/2, frequency is positive
//             and for indices larger than (N-1)/2 frequency is negative
    int n;
    double df = S/N;
    if (f < 0)
        n = round(f/df + N);
    else
        n = round(f/df);
    return n;

}

void isce::signal::Signal::
Filter(std::vector<float>& filter )
{  
    _filteredSpectrum = _spectrum; // ???
    for (int i=0; i< _nfft; i++)
         _filteredSpectrum[i] = _filteredSpectrum[i]*filter[i];
    
}


void isce::signal::Signal::
demodulate(float samplingRate, float subBandCenterFrequency)
{

    std::vector<float> time(_nfft);
    for (int i = 0; i < _nfft; i++)
        time[i] = i/samplingRate;

    for (int i = 0; i < _nfft; i++)
        _filteredData[i] = _filteredData[i]*(std::exp(-1.0f*I*2.0f*PI*subBandCenterFrequency*time[i]));

}


std::vector< std::complex<float> > isce::signal::Signal::
getData(){
    return _data;
}

std::vector< std::complex<float> > isce::signal::Signal::
getSpectrum(){
    return _spectrum;
}

std::vector< std::complex<float> > isce::signal::Signal::
getFilteredData(){
    return _filteredData;
}

std::vector< std::complex<float> > isce::signal::Signal::
getFilteredSpectrum(){
    return _filteredSpectrum;
}

