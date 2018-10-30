
#ifndef ISCE_SIGNAL_SIGNAL_H
#define ISCE_SIGNAL_SIGNAL_H

#include <cmath>
#include <valarray>

#include <isce/core/Constants.h>

#include "fftw3cxx.h"

// Declaration
namespace isce {
    namespace signal {
        template<class T>
        class Signal;
    }
}

template<class T> 
class isce::signal::Signal {
    public:
        /** Default constructor. */ 
        Signal() {};

        ~Signal() {};

        T fftPlanForward(std::valarray<std::complex<T>> &input, 
					std::valarray<std::complex<T>> &output,
            				int rank, int n, int howmany,
            				int inembed, int istride, int idist,
            				int onembed, int ostride, int odist, int sign);

        T fftPlanBackward(std::valarray<std::complex<T>> &input,
                                        std::valarray<std::complex<T>> &output,
                                        int rank, int n, int howmany,
                                        int inembed, int istride, int idist,
                                        int onembed, int ostride, int odist, int sign);	

        T forward(std::valarray<std::complex<T>> &input,
                                        std::valarray<std::complex<T>> &output);
        
        T inverse(std::valarray<std::complex<T>> &input,
                                        std::valarray<std::complex<T>> &output);

        T forwardRangeFFT(std::valarray<std::complex<T>>& signal, 
					std::valarray<std::complex<T>>& spectrum,
                			int incolumns, int inrows, 
                                        int outcolumns, int outrows);

        T forwardAzimuthFFT(std::valarray<std::complex<T>> &signal,
                                        std::valarray<std::complex<T>> &spectrum,
                                        int incolumns, int inrows, 
                                        int outcolumns, int outrows);

        
        T inverseRangeFFT(std::valarray<std::complex<T>> &spectrum, 
                                        std::valarray<std::complex<T>> &signal,
                                        int incolumns, int inrows, 
                                        int outcolumns, int outrows);

        T inverseAzimuthFFT(std::valarray<std::complex<T>> &spectrum,
                                        std::valarray<std::complex<T>> &signal,
                                        int incolumns, int inrows,
                                        int outcolumns, int outrows);

        T upsample(std::valarray<std::complex<T>> &signal,
                                        std::valarray<std::complex<T>> &signalOversampled,
                                        int rows, int nfft, int oversampleFactor);

        //
        //band pass filter to certain sub-bands 
        //each sub-band is specified by its center frequency and its bandwidth

        //void bandPass(double samplingRate, std::vector<double> band);

        //void bandPass(double samplingRate, double subBandCenterFrequency, double subBandBandwidth);

        // Filter the spectrum in frequency domain
        //void applyFilter(std::vector<float>&);

        // baseband the signal 
        //void demodulate(float samplingRate, float subBandCenterFrequency);

    private:
        int _nfft;
        int _oversample;
        isce::fftw3cxx::plan<T> _plan_fwd;
        isce::fftw3cxx::plan<T> _plan_inv;
        // data members 
        std::vector<float>  _frequencies;
        int _indexOfFrequency(double S, int N, double f);

};

#endif


