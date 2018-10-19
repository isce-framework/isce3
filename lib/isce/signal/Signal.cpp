#include "Signal.h"
#include <iostream>

isce::signal::Signal::
~Signal() {
    //fftwf_destroy_plan(_plan_fwd);
    //fftwf_destroy_plan(_plan_inv);
}

template<typename T>
void isce::signal::Signal::
forwardFFT_1D(std::valarray<std::complex<T>> &signal, std::valarray<std::complex<T>> &spectrum, size_t N)
{

    isce::fftw3cxx::plan<T> _plan;
    _plan = fftw3cxx::plan<T>::plan_dft_1d(N, 
                                            &signal[0], 
                                            &spectrum[0], 
                                            FFTW_FORWARD, FFTW_ESTIMATE);
    _plan.execute();

}

template<typename T>
void isce::signal::Signal::
forwardFFT(std::valarray<std::complex<T>> &signal, std::valarray<std::complex<T>> &spectrum, 
	    int rank, int n, int howmany,
            int inembed, int istride, int idist,
            int onembed, int ostride, int odist)
{

    isce::fftw3cxx::plan<T> _plan;
    _plan = fftw3cxx::plan<T>::plan_many_dft(rank, &n, howmany,
                                            &signal[0], &inembed, istride, idist,
                                            &spectrum[0], &onembed, ostride, odist,
                                            FFTW_FORWARD, FFTW_ESTIMATE);
    _plan.execute();

}

template<typename T>
void isce::signal::Signal::
forwardRangeFFT(std::valarray<std::complex<T>> &signal, std::valarray<std::complex<T>> &spectrum,
                int incolumns, int inrows, int outcolumns, int outrows)
{
    int rank = 1;
    int n = incolumns;
    int howmany = inrows;
    int inembed = incolumns;
    int istride = 1;
    int idist = incolumns;

    int onembed = outcolumns;
    int ostride = 1;
    int odist = outcolumns;

    forwardFFT(signal, spectrum, rank, n, howmany,
                inembed, istride, idist,
                onembed, ostride, odist);

}

/*
template<typename T>
void isce::signal::Signal::
forwardAzimuthFFT(std::valarray<std::complex<T>> &signal, std::valarray<std::complex<T>> &spectrum,
                        int incolumns, int inrows, outcolumns, outrows)
{
        in
}
*/

// need to decalre each template function specifically to make them visible to compiler
// We currently allow float and double. If at any time "long double" is needed we should add it here. 

template void isce::signal::Signal::
forwardFFT_1D(std::valarray<std::complex<float>> &signal, 
                std::valarray<std::complex<float>> &spectrum, 
                size_t N);
template void isce::signal::Signal::
forwardFFT_1D(std::valarray<std::complex<double>> &signal, 
            std::valarray<std::complex<double>> &spectrum, 
            size_t N);


template void isce::signal::Signal::
forwardFFT(std::valarray<std::complex<float>> &signal, 
            std::valarray<std::complex<float>> &spectrum,
            int rank, int n, int howmany,
            int inembed, int istride, int idist,
            int onembed, int ostride, int odist);

template void isce::signal::Signal::
forwardFFT(std::valarray<std::complex<double>> &signal,
            std::valarray<std::complex<double>> &spectrum,
            int rank, int n, int howmany,
            int inembed, int istride, int idist,
            int onembed, int ostride, int odist);


template void isce::signal::Signal::
forwardRangeFFT(std::valarray<std::complex<float>> &signal,
                std::valarray<std::complex<float>> &spectrum,
                int incolumns, int inrows, int outcolumns,int outrows);

template void isce::signal::Signal::
forwardRangeFFT(std::valarray<std::complex<double>> &signal, 
                std::valarray<std::complex<double>> &spectrum,
                int incolumns, int inrows, int outcolumns,int outrows);


