#include "Signal.h"
#include <iostream>

isce::signal::Signal::
~Signal() {}


template<typename T>
void isce::signal::Signal::
FFT(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output, 
	    int rank, int n, int howmany,
            int inembed, int istride, int idist,
            int onembed, int ostride, int odist, int sign)
{

    isce::fftw3cxx::plan<T> _plan;
    _plan = fftw3cxx::plan<T>::plan_many_dft(rank, &n, howmany,
                                            &input[0], &inembed, istride, idist,
                                            &output[0], &onembed, ostride, odist,
                                            sign, FFTW_ESTIMATE);
    _plan.execute();

}

template<typename T>
void isce::signal::Signal::
forwardRangeFFT(std::valarray<std::complex<T>> &signal, std::valarray<std::complex<T>> &spectrum,
                int incolumns, int inrows, int outcolumns, int outrows)
{
    int rank = 1;
    int n = outcolumns;
    int howmany = inrows;
    int inembed = incolumns;
    int istride = 1;
    int idist = incolumns;

    int onembed = outcolumns;
    int ostride = 1;
    int odist = outcolumns;

    FFT(signal, spectrum, rank, n, howmany,
                inembed, istride, idist,
                onembed, ostride, odist, FFTW_FORWARD);

}


template<typename T>
void isce::signal::Signal::
forwardAzimuthFFT(std::valarray<std::complex<T>> &signal, 
                std::valarray<std::complex<T>> &spectrum,
                int incolumns, int inrows, int outcolumns, int outrows)
{

    int rank = 1;
    int n = outrows;
    int howmany = incolumns;
    int inembed = inrows;
    int istride = incolumns;
    int idist = 1;

    int onembed = outrows;
    int ostride = outcolumns;
    int odist = 1;

    FFT(signal, spectrum, rank, n, howmany,
                inembed, istride, idist,
               onembed, ostride, odist, FFTW_FORWARD);

}

template<typename T>
void isce::signal::Signal::
inverseRangeFFT(std::valarray<std::complex<T>> &spectrum, std::valarray<std::complex<T>> &signal,
                int incolumns, int inrows, int outcolumns, int outrows)
{
    int rank = 1;
    int n = outcolumns;
    int howmany = inrows;
    int inembed = incolumns;
    int istride = 1;
    int idist = incolumns;

    int onembed = outcolumns;
    int ostride = 1;
    int odist = outcolumns;

    FFT(spectrum, signal, rank, n, howmany,
                inembed, istride, idist,
                onembed, ostride, odist, FFTW_BACKWARD);
    signal /=n;
}

template<typename T>
void isce::signal::Signal::
inverseAzimuthFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<std::complex<T>> &signal,
                int incolumns, int inrows, int outcolumns, int outrows)
{

    int rank = 1;
    int n = outrows;
    int howmany = incolumns;
    int inembed = inrows;
    int istride = incolumns;
    int idist = 1;

    int onembed = outrows;
    int ostride = outcolumns;
    int odist = 1;

    FFT(spectrum, signal, rank, n, howmany,
                inembed, istride, idist,
               onembed, ostride, odist, FFTW_BACKWARD);
    signal/=n;
}

// need to decalre each template function specifically to make them visible to compiler
// We currently allow float and double. If at any time "long double" is needed we should add it here. 


template void isce::signal::Signal::
FFT(std::valarray<std::complex<float>> &input, 
            std::valarray<std::complex<float>> &output,
            int rank, int n, int howmany,
            int inembed, int istride, int idist,
            int onembed, int ostride, int odist, int sign);

template void isce::signal::Signal::
FFT(std::valarray<std::complex<double>> &input,
            std::valarray<std::complex<double>> &output,
            int rank, int n, int howmany,
            int inembed, int istride, int idist,
            int onembed, int ostride, int odist, int sign);


template void isce::signal::Signal::
forwardRangeFFT(std::valarray<std::complex<float>> &signal,
                std::valarray<std::complex<float>> &spectrum,
                int incolumns, int inrows, int outcolumns,int outrows);

template void isce::signal::Signal::
forwardRangeFFT(std::valarray<std::complex<double>> &signal, 
                std::valarray<std::complex<double>> &spectrum,
                int incolumns, int inrows, int outcolumns,int outrows);

template void isce::signal::Signal::
forwardAzimuthFFT(std::valarray<std::complex<float>> &signal,
                std::valarray<std::complex<float>> &spectrum,
                int incolumns, int inrows, int outcolumns,int outrows);

template void isce::signal::Signal::
forwardAzimuthFFT(std::valarray<std::complex<double>> &signal,
                std::valarray<std::complex<double>> &spectrum,
                int incolumns, int inrows, int outcolumns,int outrows);

template void isce::signal::Signal::
inverseRangeFFT(std::valarray<std::complex<float>> &spectrum, 
		std::valarray<std::complex<float>> &signal,
                int incolumns, int inrows, int outcolumns, int outrows);

template void isce::signal::Signal::
inverseRangeFFT(std::valarray<std::complex<double>> &spectrum,
                std::valarray<std::complex<double>> &signal,
                int incolumns, int inrows, int outcolumns, int outrows);

template void isce::signal::Signal::
inverseAzimuthFFT(std::valarray<std::complex<float>> &spectrum,
                std::valarray<std::complex<float>> &signal,
                int incolumns, int inrows, int outcolumns, int outrows);

template void isce::signal::Signal::
inverseAzimuthFFT(std::valarray<std::complex<double>> &spectrum,
                std::valarray<std::complex<double>> &signal,
                int incolumns, int inrows, int outcolumns, int outrows);
