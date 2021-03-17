#include "Signal.h"
#include <iostream>
#include "fftw3cxx.h"

template<class T>
struct isce3::signal::Signal<T>::impl {
    isce3::fftw3cxx::plan<T> _plan_fwd;
    isce3::fftw3cxx::plan<T> _plan_inv;
};

template <class T>
isce3::signal::Signal<T>::
Signal() : pimpl(new impl, [](impl* p) { delete p; }) {}

template <class T>
isce3::signal::Signal<T>::
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
isce3::signal::Signal<T>::
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
isce3::signal::Signal<T>::
fftPlanForward(std::complex<T> *input, std::complex<T> *output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist, int sign)
{

    _fwd_configure(rank, n, howmany, 
               inembed, istride, idist, 
               onembed, ostride, odist);

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
isce3::signal::Signal<T>::
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
isce3::signal::Signal<T>::
fftPlanForward(T *input, std::complex<T> *output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist)
{

    _fwd_configure(rank, n, howmany, 
               inembed, istride, idist, 
               onembed, ostride, odist);

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
isce3::signal::Signal<T>::
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
isce3::signal::Signal<T>::
fftPlanBackward(std::complex<T> *input, std::complex<T> *output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist, int sign)
{

    _rev_configure(rank, n, howmany, 
               inembed, istride, idist, 
               onembed, ostride, odist);

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
isce3::signal::Signal<T>::
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
isce3::signal::Signal<T>::
fftPlanBackward(std::complex<T> *input, T *output,
            int rank, int *n, int howmany,
            int *inembed, int istride, int idist,
            int *onembed, int ostride, int odist)
{

    _rev_configure(rank, n, howmany, 
               inembed, istride, idist, 
               onembed, ostride, odist);

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
isce3::signal::Signal<T>::
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
isce3::signal::Signal<T>::
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
isce3::signal::Signal<T>::
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
isce3::signal::Signal<T>::
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
isce3::signal::Signal<T>::
inverse(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    pimpl->_plan_inv.execute_dft(&input[0], &output[0]);
}

/** unnormalized inverse transform.*/
template<class T>
void
isce3::signal::Signal<T>::
inverse(std::complex<T> *input, std::complex<T> *output)
{
    pimpl->_plan_inv.execute_dft(input, output);
}

/** unnormalized inverse transform.*/
template<class T>
void
isce3::signal::Signal<T>::
inverse(std::valarray<std::complex<T>> &input, std::valarray<T> &output)
{
    pimpl->_plan_inv.execute_dft_c2r(&input[0], &output[0]);
}

/** unnormalized inverse transform.*/
template<class T>
void
isce3::signal::Signal<T>::
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
void isce3::signal::Signal<T>::
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
void isce3::signal::Signal<T>::
forwardRangeFFT(std::complex<T> *signal,
                std::complex<T> *spectrum,
                int ncolumns, int nrows)
{

    _fwd_configureRangeFFT(ncolumns, nrows);
    
    fftPlanForward(signal, spectrum, _fwd_rank, _fwd_n, _fwd_howmany,
                    _fwd_inembed, _fwd_istride, _fwd_idist,
                    _fwd_onembed, _fwd_ostride, _fwd_odist, FFTW_FORWARD);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void isce3::signal::Signal<T>::
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
void isce3::signal::Signal<T>::
forwardRangeFFT(T *signal,
                std::complex<T> *spectrum,
                int ncolumns, int nrows)
{

    _fwd_configureRangeFFT(ncolumns, nrows);   

    fftPlanForward(signal, spectrum, _fwd_rank, _fwd_n, _fwd_howmany,
                _fwd_inembed, _fwd_istride, _fwd_idist,
                _fwd_onembed, _fwd_ostride, _fwd_odist);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
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
void isce3::signal::Signal<T>::
forwardAzimuthFFT(std::complex<T> *signal,
                std::complex<T> *spectrum,
                int ncolumns, int nrows)
{

    _fwd_configureAzimuthFFT(ncolumns, nrows);

    fftPlanForward(signal, spectrum, _fwd_rank, _fwd_n, _fwd_howmany,
                _fwd_inembed, _fwd_istride, _fwd_idist,
                _fwd_onembed, _fwd_ostride, _fwd_odist, FFTW_FORWARD);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void isce3::signal::Signal<T>::
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
void isce3::signal::Signal<T>::
forwardAzimuthFFT(T *signal,
                std::complex<T> *spectrum,
                int ncolumns, int nrows)
{

    _fwd_configureAzimuthFFT(ncolumns, nrows);

    fftPlanForward(signal, spectrum, _fwd_rank, _fwd_n, _fwd_howmany,
                   _fwd_inembed, _fwd_istride, _fwd_idist,
                   _fwd_onembed, _fwd_ostride, _fwd_odist);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
forward2DFFT(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum,
                int ncolumns, int nrows)
{

    forward2DFFT(&signal[0], &spectrum[0],
                 ncolumns, nrows, ncolumns, nrows);
}


/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] incolumns number of columns of the block of data
*  @param[in] inrows number of rows of the block of data
*  @param[in] oncolumns number of columns of the block of data in output container
*  @param[in] onrows number of rows of the block of data in output container
*
*  If doing an out-of-place FFT, the output can be stored in a container whose
*  number of columns and rows could be different than the input. This would be
*  necessary for instance in the case of 2D upsampling
*/
template<class T>
void isce3::signal::Signal<T>::
forward2DFFT(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &spectrum,
                int incolumns, int inrows,
                int oncolumns, int onrows)
{
    forward2DFFT(&signal[0], &spectrum[0],
                 incolumns, inrows, oncolumns, onrows);
}


/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
forward2DFFT(std::complex<T>* signal,
                std::complex<T>* spectrum,
                int ncolumns, int nrows)
{

    forward2DFFT(signal, spectrum, ncolumns, nrows, ncolumns, nrows);

}



/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] incolumns number of columns of the block of data in input container
*  @param[in] inrows number of rows of the block of data in input container
*  @param[in] oncolumns number of columns of the block of data in output container
*  @param[in] oupnrows number of rows of the block of data in output container
*
*  If doing an out-of-place FFT, the output can be stored in a container whose
*  number of columns and rows could be different than the input. This would be
*  necessary for instance in the case of 2D upsampling
*/
template<class T>
void isce3::signal::Signal<T>::
forward2DFFT(std::complex<T>* signal,
                std::complex<T>* spectrum,
                int incolumns, int inrows,
                int oncolumns, int onrows)
{

    _fwd_configure2DFFT(incolumns, inrows, oncolumns, onrows);

    fftPlanForward(signal, spectrum, _fwd_rank, _fwd_n, _fwd_howmany,
                    _fwd_inembed, _fwd_istride, _fwd_idist,
                    _fwd_onembed, _fwd_ostride, _fwd_odist, FFTW_FORWARD);

}



/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] incolumns number of columns of the block of data in input container
*  @param[in] inrows number of rows of the block of data in input container
*  @param[in] oncolumns number of columns of the block of data in output container
*  @param[in] oupnrows number of rows of the block of data in output container
*
*  If doing an out-of-place FFT, the output can be stored in a container whose
*  number of columns and rows could be different than the input. This would be
*  necessary for instance in the case of 2D upsampling
*/
template<class T>
void isce3::signal::Signal<T>::
forward2DFFT(std::valarray<T> &signal,
            std::valarray<std::complex<T>> &spectrum,
            int incolumns, int inrows,
            int oncolumns, int onrows)
{
    forward2DFFT(&signal[0],
            &spectrum[0],
            incolumns, inrows,
            oncolumns, onrows);
}


/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
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
void isce3::signal::Signal<T>::
forward2DFFT(T* signal,
            std::complex<T>* spectrum,
            int ncolumns, int nrows)
{
    forward2DFFT(signal, spectrum,
                 ncolumns, nrows, ncolumns, nrows);
}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] incolumns number of columns of the block of data in input container
*  @param[in] inrows number of rows of the block of data in input container
*  @param[in] oncolumns number of columns of the block of data in output container
*  @param[in] oupnrows number of rows of the block of data in output container
*
*  If doing an out-of-place FFT, the output can be stored in a container whose
*  number of columns and rows could be different than the input. This would be
*  necessary for instance in the case of 2D upsampling.
*/
template<class T>
void isce3::signal::Signal<T>::
forward2DFFT(T* signal,
            std::complex<T>* spectrum,
            int incolumns, int inrows,
            int oncolumns, int onrows)
{

    _fwd_configure2DFFT(incolumns, inrows, oncolumns, onrows);

    fftPlanForward(signal, spectrum, _fwd_rank, _fwd_n, _fwd_howmany,
                    _fwd_inembed, _fwd_istride, _fwd_idist,
                    _fwd_onembed, _fwd_ostride, _fwd_odist);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverseRangeFFT(std::complex<T>* spectrum, 
                std::complex<T>* signal,
                int ncolumns, int nrows)
{
    _rev_configureRangeFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverseRangeFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<std::complex<T>> &signal,
                int ncolumns, int nrows)
{
    _rev_configureRangeFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverseRangeFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<T> &signal,
                int ncolumns, int nrows)
{
    _rev_configureRangeFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverseRangeFFT(std::complex<T>* spectrum,
                T* signal,
                int ncolumns, int nrows)
{
    _rev_configureRangeFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverseAzimuthFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<std::complex<T>> &signal,
                int ncolumns, int nrows)
{

    _rev_configureAzimuthFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverseAzimuthFFT(std::complex<T>* spectrum,
                    std::complex<T>* signal,
                    int ncolumns, int nrows)
{

    _rev_configureAzimuthFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverseAzimuthFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<T> &signal,
                int ncolumns, int nrows)
{

    _rev_configureAzimuthFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverseAzimuthFFT(std::complex<T>* spectrum,
                T* signal,
                int ncolumns, int nrows)
{

    _rev_configureAzimuthFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist);
}


/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverse2DFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<std::complex<T>> &signal,
                int ncolumns, int nrows)
{

    _rev_configure2DFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverse2DFFT(std::complex<T>* spectrum,
                std::complex<T>* signal,
                int ncolumns, int nrows)
{

    _rev_configure2DFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverse2DFFT(std::valarray<std::complex<T>> &spectrum,
                std::valarray<T> &signal,
                int ncolumns, int nrows)
{

    _rev_configure2DFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
*/
template<class T>
void isce3::signal::Signal<T>::
inverse2DFFT(std::complex<T>* spectrum,
                T* signal,
                int ncolumns, int nrows)
{

    _rev_configure2DFFT(ncolumns, nrows);

    fftPlanBackward(spectrum, signal, _rev_rank, _rev_n, _rev_howmany,
                    _rev_inembed, _rev_istride, _rev_idist,
                    _rev_onembed, _rev_ostride, _rev_odist);
}

/**
*   @param[in] signal input block of data
*   @param[in] signalUpsampled output block of oversampled data
*   @param[in] rows number of rows of the block of input and upsampled data
*   @param[in] fft_size number of columns of the block of input data
*   @param[in] upsampleFactor upsampling factor
*/
template<class T>
void isce3::signal::Signal<T>::
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
void isce3::signal::Signal<T>::
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
    for (size_t column = 0; column<(fft_size+1)/2; ++column)
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

/**
 *   @param[in] signal input block of data
 *   @param[out] signalUpsampled output block of oversampled data
 *   @param[in] shiftImpact a linear phase term equivalent to a constant shift
 *   in time domain
 */
template<class T>
void isce3::signal::Signal<T>::upsample(
        isce3::core::EArray2D<std::complex<T>>& signal,
        isce3::core::EArray2D<std::complex<T>>& signalUpsampled,
        const isce3::core::EArray2D<std::complex<T>>& shiftImpact)
{

    // number of rows which are the same before and after upsampling in range
    int nrows = signal.rows();

    // number of columns in the original signal
    int fft_size = signal.cols();

    // number of columns of upsampled data
    int columns = signalUpsampled.cols();

    // temporary storage for the spectrum before and after the shift
    isce3::core::EArray2D<std::complex<T>> spectrum(nrows, fft_size);
    isce3::core::EArray2D<std::complex<T>> spectrumShifted(nrows, columns);

    spectrumShifted = std::complex<T>(0.0, 0.0);

    // forward fft in range
    pimpl->_plan_fwd.execute_dft(signal.data(), spectrum.data());

    // spectrum /= fft_size;
    // shift the spectrum
    // The spectrum has values from begining to fft_size index for each line. We
    // want to put the spectrum in correct ouput locations such that the
    // spectrum of the sampled data has values from 0 to fft_size/2 and from
    // upsampleFactor*fft_size - fft_size/2 to the end. For a 1D example:
    //      spectrum = [1,2,3,4,5,6,0,0,0,0,0,0]
    //  becomes:
    //      spectrumShifted = [1,2,3,0,0,0,0,0,0,4,5,6]
    //

    spectrumShifted.block(0, 0, nrows, (fft_size + 1) / 2) =
            spectrum.block(0, 0, nrows, (fft_size + 1) / 2);
    spectrumShifted.block(0, columns - fft_size / 2, nrows, fft_size / 2) =
            spectrum.block(0, (fft_size + 1) / 2, nrows, fft_size / 2);

    if (shiftImpact.rows() != 0)
        spectrumShifted *= shiftImpact;

    // inverse fft to get the upsampled signal
    pimpl->_plan_inv.execute_dft(spectrumShifted.data(),
                                 signalUpsampled.data());

    // Normalize
    signalUpsampled /= fft_size;
}

/**
*   @param[in] signal input block of 2D data
*   @param[out] signalUpsampled output block of oversampled data
*   @param[in] upsampleFactor upsampling factor
*
*   When doing out-of-place upsampling, i.e., when the container of the input to upsample is
*   different to the one that will contain the upsampled data, the FFT forward plan must be
*   set to out-of-place (with the input and output containers) and the FFT reverse plan to 
*   in-place (with the output container). 
*   In both case (in-place or out-of-place 2D upsampling), it is the user responsability to 
*   provide an output container that is padded (for in-place) or filled (for out-of-place) 
*   with zeros.
*/
template<class T>
void isce3::signal::Signal<T>::
upsample2D(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &signalUpsampled,
                int oversampleFactor) {

    upsample2D(&signal[0], &signalUpsampled[0], 
               oversampleFactor, nullptr);

}



/**
*   @param[in] signal input block of 2D data
*   @param[out] signalUpsampled output block of oversampled data
*   @param[in] upsampleFactor upsampling factor
*   @param[in] shiftImpact a linear phase term equivalent to a constant shift in time domain 
*
*   When doing out-of-place upsampling, i.e., when the container of the input to upsample is
*   different to the one that will contain the upsampled data, the FFT forward plan must be
*   set to out-of-place (with the input and output containers) and the FFT reverse plan to 
*   in-place (with the output container).
*   In both case (in-place or out-of-place 2D upsampling), it is the user responsability to 
*   provide an output container that is padded (for in-place) or filled (for out-of-place) 
*   with zeros.
*/
template<class T>
void isce3::signal::Signal<T>::
upsample2D(std::valarray<std::complex<T>> &signal,
                std::valarray<std::complex<T>> &signalUpsampled,
                int upsampleFactor,
                std::valarray<std::complex<T>> &shiftImpact) {

   upsample2D(&signal[0], &signalUpsampled[0],
              upsampleFactor, &shiftImpact[0]); 

}


/**
*   @param[in] signal input block of 2D data
*   @param[out] signalUpsampled output block of oversampled data
*   @param[in] upsampleFactor upsampling factor
*
*   When doing out-of-place upsampling, i.e., when the container of the input to upsample is
*   different to the one that will contain the upsampled data, the FFT forward plan must be
*   set to out-of-place (with the input and output containers) and the FFT reverse plan to 
*   in-place (with the output container).
*   In both case (in-place or out-of-place 2D upsampling), it is the user responsability to 
*   provide an output container that is padded (for in-place) or filled (for out-of-place) 
*   with zeros.
*/
template<class T>
void isce3::signal::Signal<T>::
upsample2D(std::complex<T> *signal,
           std::complex<T> *signalUpsampled,
           int upsampleFactor) 
{
   upsample2D(signal, signalUpsampled,
              upsampleFactor, nullptr); 
}



/**
*   @param[in] signal input block of 2D data
*   @param[out] signalUpsampled output block of oversampled data
*   @param[in] upsampleFactor upsampling factor
*   @param[in] shiftImpact a linear phase term equivalent to a constant shift in time domain 
*
*   When doing out-of-place upsampling, i.e., when the container of the input to upsample is
*   different to the one that will contain the upsampled data, the FFT forward plan must be
*   set to out-of-place (with the input and output containers) and the FFT reverse plan to 
*   in-place (with the output container).
*   In both case (in-place or out-of-place 2D upsampling), it is the user responsability to 
*   provide an output container that is padded (for in-place) or filled (for out-of-place) 
*   with zeros.
*/
template<class T>
void isce3::signal::Signal<T>::
upsample2D(std::complex<T> *signal,
           std::complex<T> *signalUpsampled,
           int upsampleFactor, 
           std::complex<T> *shiftImpact)
{
    
    //rank 2 is the only possible value. Just as a sanity check
    if (_fwd_rank != 2 || _rev_rank !=2) { 
       std::cout << "upsample2D required rank 2 inputs" << std::endl;
       return;
    }
   
    // Sanity check that reverse FFT rank is coherent with upsampleFactor
    if (upsampleFactor * _fwd_n[0] != _rev_n[0] || upsampleFactor * _fwd_n[1] != _rev_n[1]) {
        std::cout << "Discrepency between upsampling factor and reverse FFT rank values" << std::endl;
        return;
    }
    


    // Spectrum shift
    // Dimensions of the quarts
    //
    //        cols1  cols2
    //       _____________ 
    //       |     |      |
    // rows1 |     |      |
    //       |_____|______|
    //       |     |      |
    // rows2 |     |      |
    //       |_____|______| 
    //
    //
    //
    size_t cols2 = _fwd_n[0]/2;
    size_t cols1 = _fwd_n[0] - cols2;
    size_t rows2 = _fwd_n[1]/2;
    size_t rows1 = _fwd_n[1] - rows2;


    // [1] Forward fft
    // If doing out-of-place transform, i.e., input signal is a different container than
    // output container, the forward FFT is done out-of-place and the reverse FFT will be
    // done in-place.
    if (signal != signalUpsampled) 
       pimpl->_plan_fwd.execute_dft(signal, signalUpsampled);
    else
       pimpl->_plan_fwd.execute_dft(signalUpsampled, signalUpsampled);


    // [2] Spectrum shuffling - Moving the 4 quarts to the corners of the output (larger)
    // container. In-place operation
    #pragma omp parallel for
    for (size_t p = 0; p < _fwd_howmany; p++) {

        std::complex<T> * loc;

        for (size_t r = 0; r < rows1; r++) {
            loc = &signalUpsampled[p * _fwd_odist + r * _fwd_onembed[0] + cols1];
            std::swap_ranges(loc, 
                             loc + cols2,
                             &signalUpsampled[p * _rev_idist + r * _rev_onembed[0] + _rev_n[0] - cols2]);
        }

        for (size_t r = 0; r < rows2; r++) {
            loc = &signalUpsampled[p * _fwd_odist + (r+rows1) * _fwd_onembed[0]];
            std::swap_ranges(loc, 
                             loc+cols1,
                             &signalUpsampled[p * _rev_idist + (_rev_n[1] - rows2 + r) * _rev_onembed[0]]);
            loc = &signalUpsampled[p * _fwd_odist + (r+rows1) * _fwd_onembed[0] + cols1];
            std::swap_ranges(loc, 
                             loc + cols2,
                             &signalUpsampled[p * _rev_idist + (_rev_n[1] - rows2 + 1 + r) * _rev_onembed[0] - cols2]);
        }

    }    


    // multiply the shiftImpact (a linear phase is frequency domain
    // equivalent to a shift in time domain) by the spectrum
    if (shiftImpact != nullptr) {
       #pragma omp parallel for
       for (size_t p = 0; p < _fwd_howmany; p++) 
          for (size_t j = 0; j < _rev_n[1]; j++) 
             for (size_t i = 0; i < _rev_n[0]; i++) 
                signalUpsampled[p*_rev_idist + j*_rev_onembed[0] + i] *= shiftImpact[j*_rev_n[0] + i];
    }


    // [3] Inverse fft to get the upsampled signal
    pimpl->_plan_inv.execute_dft(signalUpsampled, signalUpsampled);


    // [4] Normalize
    size_t sz = _fwd_n[0] * _fwd_n[1];
    #pragma omp parallel for
    for (size_t p = 0; p < _fwd_howmany; p++) 
       for (size_t j = 0; j < _rev_n[1]; j++) 
          for (size_t i = 0; i < _rev_n[0]; i++) 
             signalUpsampled[p*_rev_idist + j*_rev_onembed[0] + i] /= sz;


}




// We currently allow float and double. If at any time "long double" is needed, 
// declaration should be added here. 

template class isce3::signal::Signal<float>;
template class isce3::signal::Signal<double>;

//end of file
