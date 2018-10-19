
#ifndef ISCE_SIGNAL_SIGNAL_H
#define ISCE_SIGNAL_SIGNAL_H

// std
#include <cmath>
#include <valarray>

//fftw
//#include <fftw3.h>
// isce::core
#include <isce/core/Constants.h>

#include "fftw3cxx.h"

//template <typename T>
//const std::complex<T> I(0, 1);

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

        template<typename T> void forwardFFT_1D(std::valarray<std::complex<T>>& signal, 
                                            std::valarray<std::complex<T>>& spectrum,
                                            size_t N);

        template<typename T> void forwardFFT(std::valarray<std::complex<T>> &signal, 
					std::valarray<std::complex<T>> &spectrum,
            				int rank, int n, int howmany,
            				int inembed, int istride, int idist,
            				int onembed, int ostride, int odist);
	
	
        template<typename T> void forwardRangeFFT(std::valarray<std::complex<T>>& signal, 
					std::valarray<std::complex<T>>& spectrum,
                			int incolumns, int inrows, 
                                        int outcolumns, int outrows);

        template<typename T> void forwardAzimuthFFT(std::valarray<std::complex<T>> &signal,
                                                    std::valarray<std::complex<T>> &spectrum,
                                                    int incolumns, int inrows, 
                                                    int outcolumns, int outrows);

        //template<typename T> void forwardFFT(std::valarray<std::complex<T>>& signal,
        //                                        std::valarray<std::complex<T>>& spectrum,
        //                                        size_t N);
        

        //template<typename T> void inverseFFT(std::valarray<T>& spectrum, std::valarray<T>& signal);

        //template<typename T> void rangeFFT(std::valarray<T>& signal, std::valarray<T>& spectrum);
        //template<typename T> void azimuthFFT(std::valarray<T>& signal, std::valarray<T>& spectrum);

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

        // data members 
        std::vector<float>  _frequencies;
        int _indexOfFrequency(double S, int N, double f);

};

#endif


