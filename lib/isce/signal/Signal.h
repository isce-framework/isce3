
#ifndef ISCE_SIGNAL_SIGNAL_H
#define ISCE_SIGNAL_SIGNAL_H

// std
#include <cmath>
#include <valarray>

//fftw
#include <fftw3.h>

// isce::core
#include <isce/core/Constants.h>

const std::complex<float> I(0, 1);
const float PI = std::acos(-1);
// Declaration
namespace isce {
    namespace signal {
        class Signal;
    }
}


class isce::signal::Signal {
    public:
        /** Default constructor. */ 
        Signal() {};

        ~Signal();

        Signal(std::vector<std::complex<float>>& data, int oversample);
        
        void initFFTPlans();
        // forward 1D fft

        void forwardFFT() { fftwf_execute(_plan_fwd);}

        // inverse 1D fftw
        void inverseFFT() { fftwf_execute(_plan_inv); }

        //band pass filter to certain sub-bands 
        //each sub-band is specified by its center frequency and its bandwidth
        void bandPass(double samplingRate, std::vector<double> Band);

        void bandPass(double samplingRate, double subBandCenterFrequency, double subBandBandwidth);

        // Filter the spectrum in frequency domain
        void Filter(std::vector<float>&);

        // baseband the signal 
        void demodulate(float samplingRate, float subBandCenterFrequency);

        std::vector< std::complex<float> > getData();
        std::vector< std::complex<float> > getSpectrum();
        std::vector< std::complex<float> > getFilteredData();
        std::vector< std::complex<float> > getFilteredSpectrum();

    private:
        int _nfft;
        int _oversample;

        // data members 
        std::vector< std::complex<float> > _data;
        std::vector< std::complex<float> > _spectrum;
        std::vector< std::complex<float> > _filteredData;
        std::vector< std::complex<float> > _filteredSpectrum;
        std::vector<float>  _frequencies;

        // FFTW plans
        fftwf_plan _plan_fwd, _plan_inv;
        int _indexOfFrequency(double S, int N, double f);

};

#endif


