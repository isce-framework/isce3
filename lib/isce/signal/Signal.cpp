// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#include "Signal.h"
#include <iostream>
#include "fftw3cxx.h"

template<class T>
struct isce::signal::Signal<T>::impl {
    isce::fftw3cxx::plan<T> _plan_fwd;
    isce::fftw3cxx::plan<T> _plan_inv;
};

template <class T>
isce::signal::Signal<T>::
Signal() : pimpl(new impl, [](impl* p) { delete p; }) {}

template <class T>
isce::signal::Signal<T>::
Signal(int nthreads) : pimpl(new impl, [](impl* p) { delete p; }) {
    fftw3cxx::init_threads<T>();
    fftw3cxx::plan_with_nthreads<T>(nthreads);
}

/**
*  @param[in] input block of data
*  @param[out] output block of data
*  @param[in] rank rank of the transform (1: for one dimensional and 2: for two dimensional transform)
*  @param[in] size size of each transform (ncols: for range FFT, nrows: for azimuth FFT)
*  @param[in] howmany number of FFT transforms for a block of data (nrows: for range FFT, ncols: for azimuth FFT)
*  @param[in] inembed 
*  @param[in] istride
*  @param[in] idist
*  @param[in] onembed
*  @param[in] ostride
*  @param[in] odist
*  @param[in] sign
*/
template <class T>
void 
isce::signal::Signal<T>::
fftPlanForward(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output, 
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist, int sign)
{

    fftPlanForward(&input[0], &output[0],
            rank, n, howmany,
            inembed, istride, idist,
            onembed, ostride, odist, sign);

}

/**
*  @param[in] input block of data
*  @param[out] output block of data
*  @param[in] rank rank of the transform (1: for one dimensional and 2: for two dimensional transform)
*  @param[in] size size of each transform (ncols: for range FFT, nrows: for azimuth FFT)
*  @param[in] howmany number of FFT transforms for a block of data (nrows: for range FFT, ncols: for azimuth FFT)
*  @param[in] inembed
*  @param[in] istride
*  @param[in] idist
*  @param[in] onembed
*  @param[in] ostride
*  @param[in] odist
*  @param[in] sign
*/
template <class T>
void
isce::signal::Signal<T>::
fftPlanForward(std::complex<T> *input, std::complex<T> *output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist, int sign)
{

    pimpl->_plan_fwd = fftw3cxx::plan<T>::plan_many_dft(rank, n, howmany,
                                            input, inembed, istride, idist,
                                            output, onembed, ostride, odist,
                                            sign, FFTW_ESTIMATE);

}

/**
*  @param[in] input block of data
*  @param[out] output block of data
*  @param[in] rank rank of the transform (1: for one dimensional and 2: for two dimensional transform)
*  @param[in] size size of each transform (ncols: for range FFT, nrows: for azimuth FFT)
*  @param[in] howmany number of FFT transforms for a block of data (nrows: for range FFT, ncols: for azimuth FFT)
*  @param[in] inembed
*  @param[in] istride
*  @param[in] idist
*  @param[in] onembed
*  @param[in] ostride
*  @param[in] odist
*/
template <class T>
void
isce::signal::Signal<T>::
fftPlanForward(std::valarray<T> &input, std::valarray<std::complex<T>> &output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist)
{
    fftPlanForward(&input[0], &output[0],
            rank, n, howmany,
            inembed, istride, idist,
            onembed, ostride, odist);      
}

/**
*  @param[in] input block of data
*  @param[out] output block of data
*  @param[in] rank rank of the transform (1: for one dimensional and 2: for two dimensional transform)
*  @param[in] size size of each transform (ncols: for range FFT, nrows: for azimuth FFT)
*  @param[in] howmany number of FFT transforms for a block of data (nrows: for range FFT, ncols: for azimuth FFT)
*  @param[in] inembed
*  @param[in] istride
*  @param[in] idist
*  @param[in] onembed
*  @param[in] ostride
*  @param[in] odist
*/
template <class T>
void
isce::signal::Signal<T>::
fftPlanForward(T *input, std::complex<T> *output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist)
{

    pimpl->_plan_fwd = fftw3cxx::plan<T>::plan_many_dft_r2c(rank, n, howmany,
                                            input, inembed, istride, idist,
                                            output, onembed, ostride, odist,
                                            FFTW_ESTIMATE);

}


/**
*  @param[in] input block of data
*  @param[out] output block of data
*  @param[in] rank
*  @param[in] size
*  @param[in] howmany
*  @param[in] inembed
*  @param[in] istride
*  @param[in] idist
*  @param[in] onembed
*  @param[in] ostride
*  @param[in] odist
*  @param[in] sign
*/
template<class T>
void 
isce::signal::Signal<T>::
fftPlanBackward(std::valarray<std::complex<T>> &input, 
                std::valarray<std::complex<T>> &output,
                int rank, int *n, int howmany,
                int *inembed, int istride, int idist,
                int *onembed, int ostride, int odist, int sign)
{

    fftPlanBackward(&input[0], &output[0],
            rank, n, howmany,
            inembed, istride, idist,
            onembed, ostride, odist, sign);

}

/**
*  @param[in] input block of data
*  @param[out] output block of data
*  @param[in] rank
*  @param[in] size
*  @param[in] howmany
*  @param[in] inembed
*  @param[in] istride
*  @param[in] idist
*  @param[in] onembed
*  @param[in] ostride
*  @param[in] odist
*  @param[in] sign
*/
template<class T>
void
isce::signal::Signal<T>::
fftPlanBackward(std::complex<T> *input, std::complex<T> *output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist, int sign)
{

    pimpl->_plan_inv = fftw3cxx::plan<T>::plan_many_dft(rank, n, howmany,
                                            input, inembed, istride, idist,
                                            output, onembed, ostride, odist,
                                            sign, FFTW_ESTIMATE);

}

/**
*  @param[in] input block of data
*  @param[out] output block of data
*  @param[in] rank
*  @param[in] size
*  @param[in] howmany
*  @param[in] inembed
*  @param[in] istride
*  @param[in] idist
*  @param[in] onembed
*  @param[in] ostride
*  @param[in] odist
*/
template<class T>
void
isce::signal::Signal<T>::
fftPlanBackward(std::valarray<std::complex<T>> &input, std::valarray<T> &output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist)
{

    fftPlanBackward(&input[0], &output[0],
            rank, n, howmany,
            inembed, istride, idist,
            onembed, ostride, odist);

}

/**
*  @param[in] input block of data
*  @param[out] output block of data
*  @param[in] rank
*  @param[in] size
*  @param[in] howmany
*  @param[in] inembed
*  @param[in] istride
*  @param[in] idist
*  @param[in] onembed
*  @param[in] ostride
*  @param[in] odist
*/
template<class T>
void
isce::signal::Signal<T>::
fftPlanBackward(std::complex<T> *input, T *output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist)
{

    pimpl->_plan_inv = fftw3cxx::plan<T>::plan_many_dft_c2r(rank, n, howmany,
                                            input, inembed, istride, idist,
                                            output, onembed, ostride, odist,
                                            FFTW_ESTIMATE);

}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void
isce::signal::Signal<T>::
forward(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    pimpl->_plan_fwd.execute_dft(&input[0], &output[0]);
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void
isce::signal::Signal<T>::
forward(std::complex<T> *input, std::complex<T> *output)
{
    pimpl->_plan_fwd.execute_dft(input, output);
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void
isce::signal::Signal<T>::
forward(std::valarray<T> &input, std::valarray<std::complex<T>> &output)
{
    pimpl->_plan_fwd.execute_dft_r2c(&input[0], &output[0]);
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void
isce::signal::Signal<T>::
forward(T *input, std::complex<T> *output)
{
    pimpl->_plan_fwd.execute_dft_r2c(input, output);
}


/** unnormalized inverse transform. 
* Note that since the FFTW library does not
* normalize the DFT computations, computing a forward 
* followed by a backward transform (or vice versa) results 
* in the original array scaled by length of fft.
*  @param[in] input block of spectrum
*  @param[out] output block of data
*/
template<class T>
void
isce::signal::Signal<T>::
inverse(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    pimpl->_plan_inv.execute_dft(&input[0], &output[0]);
}

/** unnormalized inverse transform.*/
template<class T>
void
isce::signal::Signal<T>::
inverse(std::complex<T> *input, std::complex<T> *output)
{
    pimpl->_plan_inv.execute_dft(input, output);
}

/** unnormalized inverse transform.*/
template<class T>
void
isce::signal::Signal<T>::
inverse(std::valarray<std::complex<T>> &input, std::valarray<T> &output)
{
    pimpl->_plan_inv.execute_dft_c2r(&input[0], &output[0]);
}

/** unnormalized inverse transform.*/
template<class T>
void
isce::signal::Signal<T>::
inverse(std::complex<T> *input, T *output)
{
    pimpl->_plan_inv.execute_dft_c2r(input, output);
}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void isce::signal::Signal<T>::
forwardRangeFFT(std::valarray<std::complex<T>> &signal, 
                std::valarray<std::complex<T>> &spectrum,
                int ncolumns, int nrows)
{

    forwardRangeFFT(&signal[0],
                    &spectrum[0],
                    ncolumns, nrows);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void isce::signal::Signal<T>::
forwardRangeFFT(std::complex<T> *signal,
                std::complex<T> *spectrum,
                int ncolumns, int nrows)
{

    _configureRangeFFT(ncolumns, nrows);
    
    fftPlanForward(signal, spectrum, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist, FFTW_FORWARD);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void isce::signal::Signal<T>::
forwardRangeFFT(std::valarray<T> &signal,
                std::valarray<std::complex<T>> &spectrum,
                int ncolumns, int nrows)
{
    forwardRangeFFT(&signal[0],
                    &spectrum[0],
                    ncolumns, nrows);
}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void isce::signal::Signal<T>::
forwardRangeFFT(T *signal,
                std::complex<T> *spectrum,
                int ncolumns, int nrows)
{

    _configureRangeFFT(ncolumns, nrows);   

    fftPlanForward(signal, spectrum, _rank, _n, _howmany,
                _inembed, _istride, _idist,
                _onembed, _ostride, _odist);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
forwardAzimuthFFT(std::valarray<std::complex<T>> &signal, 
                std::valarray<std::complex<T>> &spectrum,
                int ncolumns, int nrows)
{

    forwardAzimuthFFT(&signal[0], &spectrum[0],
                    ncolumns, nrows);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void isce::signal::Signal<T>::
forwardAzimuthFFT(std::complex<T> *signal,
                std::complex<T> *spectrum,
                int ncolumns, int nrows)
{

    _configureAzimuthFFT(ncolumns, nrows);

    fftPlanForward(signal, spectrum, _rank, _n, _howmany,
                _inembed, _istride, _idist,
                _onembed, _ostride, _odist, FFTW_FORWARD);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void isce::signal::Signal<T>::
forwardAzimuthFFT(std::valarray<T> &signal,
                std::valarray<std::complex<T>> &spectrum,
                int ncolumns, int nrows)
{
    forwardAzimuthFFT(&signal[0], &spectrum[0],
                ncolumns, nrows);   
}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void isce::signal::Signal<T>::
forwardAzimuthFFT(T *signal,
                std::complex<T> *spectrum,
                int ncolumns, int nrows)
{

    _configureAzimuthFFT(ncolumns, nrows);

    fftPlanForward(signal, spectrum, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
forward2DFFT(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum,
                int ncolumns, int nrows)
{
    forward2DFFT(&signal[0], &spectrum[0],
                ncolumns, nrows);
}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
forward2DFFT(std::complex<T>* signal,
                std::complex<T>* spectrum,
                int ncolumns, int nrows)
{

    _configure2DFFT(ncolumns, nrows);

    fftPlanForward(signal, spectrum, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist, FFTW_FORWARD);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
forward2DFFT(std::valarray<T> &signal,
            std::valarray<std::complex<T>> &spectrum,
            int ncolumns, int nrows)
{
    forward2DFFT(&signal[0],
            &spectrum[0],
            ncolumns, nrows);
}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
forward2DFFT(T* signal,
            std::complex<T>* spectrum,
            int ncolumns, int nrows)
{

    _configure2DFFT(ncolumns, nrows);

    fftPlanForward(signal, spectrum, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist);

}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverseRangeFFT(std::complex<T>* spectrum, 
                std::complex<T>* signal,
                int ncolumns, int nrows)
{
    _configureRangeFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverseRangeFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<std::complex<T>> &signal,
                int ncolumns, int nrows)
{
    _configureRangeFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverseRangeFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<T> &signal,
                int ncolumns, int nrows)
{
    _configureRangeFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverseRangeFFT(std::complex<T>* spectrum,
                T* signal,
                int ncolumns, int nrows)
{
    _configureRangeFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverseAzimuthFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<std::complex<T>> &signal,
                int ncolumns, int nrows)
{

    _configureAzimuthFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverseAzimuthFFT(std::complex<T>* spectrum,
                    std::complex<T>* signal,
                    int ncolumns, int nrows)
{

    _configureAzimuthFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverseAzimuthFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<T> &signal,
                int ncolumns, int nrows)
{

    _configureAzimuthFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverseAzimuthFFT(std::complex<T>* spectrum,
                T* signal,
                int ncolumns, int nrows)
{

    _configureAzimuthFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist);
}


/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverse2DFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<std::complex<T>> &signal,
                int ncolumns, int nrows)
{

    _configure2DFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverse2DFFT(std::complex<T>* spectrum,
                std::complex<T>* signal,
                int ncolumns, int nrows)
{

    _configure2DFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverse2DFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<T> &signal,
                int ncolumns, int nrows)
{

    _configure2DFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce::signal::Signal<T>::
inverse2DFFT(std::complex<T>* spectrum,
                T* signal,
                int ncolumns, int nrows)
{

    _configure2DFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist);
}

/**
*   @param[in] signal input block of data
*   @param[in] signalUpsampled output block of oversampled data
*   @param[in] rows number of rows of the block of input and upsampled data
*   @param[in] fft_size number of columns of the block of input data
*   @param[in] upsampleFactor upsampling factor
*/
template<class T>
void isce::signal::Signal<T>::
upsample(std::valarray<std::complex<T>> &signal,
            std::valarray<std::complex<T>> &signalUpsampled,
            int rows, int fft_size, int upsampleFactor)
{

    // a dummy zero size valarray for shiftImpacts. Using this zero size 
    // shiftImpact will be interpreted as no shift application to 
    // the upsampled signal
    std::valarray<std::complex<T>> shiftImpact(0);
    
    // actually upsampling the signal
    upsample(signal, signalUpsampled,
            rows, fft_size, upsampleFactor, shiftImpact);

}

/**
*   @param[in] signal input block of data
*   @param[out] signalUpsampled output block of oversampled data
*   @param[in] rows number of rows of the block of input and upsampled data
*   @param[in] fft_size number of columns of the block of input data
*   @param[in] upsampleFactor upsampling factor
*   @param[out] shiftImpact a linear phase term equivalent to a constant shift in time domain 
*/
template<class T>
void isce::signal::Signal<T>::
upsample(std::valarray<std::complex<T>> &signal,
            std::valarray<std::complex<T>> &signalUpsampled,
            int rows, int fft_size, int upsampleFactor, 
            std::valarray<std::complex<T>> shiftImpact)
{

    // number of columns of upsampled spectrum
    int columns = upsampleFactor*fft_size;

    // temporary storage for the spectrum before and after the shift
    std::valarray<std::complex<T>> spectrum(fft_size*rows);
    std::valarray<std::complex<T>> spectrumShifted(columns*rows);

    spectrumShifted = std::complex<T> (0.0,0.0);

    // forward fft in range
    pimpl->_plan_fwd.execute_dft(&signal[0], &spectrum[0]);

    //spectrum /= fft_size;
    //shift the spectrum
    // The spectrum has values from begining to fft_size index for each line. We want
    // to put the spectrum in correct ouput locations such that the spectrum of
    // the upsampled data has values from 0 to fft_size/2 and from upsampleFactor*fft_size - fft_size/2 to the end.
    // For a 1D example:
    //      spectrum = [1,2,3,4,5,6,0,0,0,0,0,0]
    //  becomes:
    //      spectrumShifted = [1,2,3,0,0,0,0,0,0,4,5,6]
    //

    #pragma omp parallel for
    for (size_t column = 0; column<fft_size/2; ++column)
        spectrumShifted[std::slice(column, rows, columns)] = spectrum[std::slice(column, rows, fft_size)];

    #pragma omp parallel for
    for (size_t i = 0; i<fft_size/2; ++i){
        size_t j = upsampleFactor*fft_size - fft_size/2 + i;
        spectrumShifted[std::slice(j, rows, columns)] = spectrum[std::slice(i+fft_size/2, rows, fft_size)];
    }


    // multiply the shiftImpact (a linear phase is frequency domain
    // equivalent to a shift in time domain) by the spectrum
    if (spectrumShifted.size() == shiftImpact.size())
        spectrumShifted *= shiftImpact;

    // inverse fft to get the upsampled signal
    pimpl->_plan_inv.execute_dft(&spectrumShifted[0], &signalUpsampled[0]);

    // Normalize
    signalUpsampled /= fft_size;

}

// We currently allow float and double. If at any time "long double" is needed, 
// declaration should be added here. 

template class isce::signal::Signal<float>;
template class isce::signal::Signal<double>;

//end of file
