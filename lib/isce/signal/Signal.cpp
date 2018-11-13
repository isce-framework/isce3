// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#include "Signal.h"
#include <iostream>

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
T 
isce::signal::Signal<T>::
fftPlanForward(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output, 
	    int rank, int n, int howmany,
            int inembed, int istride, int idist,
            int onembed, int ostride, int odist, int sign)
{

    _plan_fwd = fftw3cxx::plan<T>::plan_many_dft(rank, &n, howmany,
                                            &input[0], &inembed, istride, idist,
                                            &output[0], &onembed, ostride, odist,
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
T 
isce::signal::Signal<T>::
fftPlanBackward(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output,
            int rank, int n, int howmany,
            int inembed, int istride, int idist,
            int onembed, int ostride, int odist, int sign)
{

    _plan_inv = fftw3cxx::plan<T>::plan_many_dft(rank, &n, howmany,
                                            &input[0], &inembed, istride, idist,
                                            &output[0], &onembed, ostride, odist,
                                            sign, FFTW_ESTIMATE);

}

/** unnormalized forward DFT computation
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
T
isce::signal::Signal<T>::
forward(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    _plan_fwd.execute_dft(&input[0], &output[0]);
}

/** unnormalized forward DFT computation. 
* Note that since the FFTW library does not
* normalize the DFT computations, computing a forward followed by a backward transform (or vice versa) results in the original array scaled by length of fft.
*  @param[in] input block of spectrum
*  @param[out] output block of data
*/
template<class T>
T
isce::signal::Signal<T>::
inverse(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    _plan_inv.execute_dft(&input[0], &output[0]);
}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] incolumns number of columns of the input block of data
*  @param[in] inrows number of rows of the input block of data
*  @param[in] outcolumns number of columns of the output block of data (this is considered as length of fft)
*  @param[in] outrows number of rows of the output block of data
 */
template<class T>
T isce::signal::Signal<T>::
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

    fftPlanForward(signal, spectrum, rank, n, howmany,
                inembed, istride, idist,
                onembed, ostride, odist, FFTW_FORWARD);

}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] incolumns number of columns of the input block of data
*  @param[in] inrows number of rows of the input block of data
*  @param[in] outcolumns number of columns of the output block of data
*  @param[in] outrows number of rows of the output block of data (this is considered as length of fft)
*/
template<class T>
T isce::signal::Signal<T>::
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

    fftPlanForward(signal, spectrum, rank, n, howmany,
                inembed, istride, idist,
               onembed, ostride, odist, FFTW_FORWARD);

}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] incolumns number of columns of the input block of data
*  @param[in] inrows number of rows of the input block of data
*  @param[in] outcolumns number of columns of the output block of data (this is considered as length of fft)
*  @param[in] outrows number of rows of the output block of data
*/
template<class T>
T isce::signal::Signal<T>::
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

    fftPlanBackward(spectrum, signal, rank, n, howmany,
                inembed, istride, idist,
                onembed, ostride, odist, FFTW_BACKWARD);
}

/**
*  @param[in] spectrum input block of spectrum
*  @param[out] signal output block of data
*  @param[in] incolumns number of columns of the input block of data
*  @param[in] inrows number of rows of the input block of data
*  @param[in] outcolumns number of columns of the output block of data
*  @param[in] outrows number of rows of the output block of data (this is considered as length of fft)
*/
template<class T>
T isce::signal::Signal<T>::
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

    fftPlanBackward(spectrum, signal, rank, n, howmany,
                inembed, istride, idist,
               onembed, ostride, odist, FFTW_BACKWARD);
}


/**
*   @param[in] signal input block of data
*   @param[in] signalUpsampled output block of oversampled data
*   @param[in] rows number of rows of the block of input and upsampled data
*   @param[in] nfft number of columns of the block of input data
*   @param[in] upsampleFactor upsampling factor
*/
template<class T>
T isce::signal::Signal<T>::
upsample(std::valarray<std::complex<T>> &signal,
         std::valarray<std::complex<T>> &signalUpsampled,
         int rows, int nfft, int upsampleFactor)
{

    // number of columns of upsampled spectrum
    int columns = upsampleFactor*nfft;

    // temporary storage for the spectrum before and after the shift
    std::valarray<std::complex<T>> spectrum(nfft*rows);
    std::valarray<std::complex<T>> spectrumShifted(columns*rows);

    // forward fft in range
    _plan_fwd.execute_dft(&signal[0], &spectrum[0]);

    //spectrum /=nfft;
    //shift the spectrum
    // The spectrum has values from begining to nfft index for each line. We want
    // to put the spectrum in correct ouput locations such that the spectrum of 
    // the upsampled data has values from 0 to nfft/2 and from upsampleFactor*nfft - nfft/2 to the end.
    // For a 1D example:
    //      spectrum = [1,2,3,4,5,6,0,0,0,0,0,0]
    //  becomes:
    //      spectrumShifted = [1,2,3,0,0,0,0,0,0,4,5,6]
    //
    
    for (size_t column = 0; column<nfft/2 - 1; ++column)
        spectrumShifted[std::slice(column, rows, columns)] = spectrum[std::slice(column, rows, nfft)];
   
    for (size_t i = 0; i<nfft/2; ++i){
        size_t j = upsampleFactor*nfft - nfft/2 + i;
        spectrumShifted[std::slice(j, rows, columns)] = spectrum[std::slice(i+nfft/2, rows, nfft)];
    }
    

    // inverse fft to get the upsampled signal
    _plan_inv.execute_dft(&spectrumShifted[0], &signalUpsampled[0]);

    // Normalize
    signalUpsampled /=nfft;

}


// We currently allow float and double. If at any time "long double" is needed, 
// declaration should be added here. 

template class isce::signal::Signal<float>;
template class isce::signal::Signal<double>;

//end of file
