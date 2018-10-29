#include "Signal.h"
#include <iostream>


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

template<class T>
T
isce::signal::Signal<T>::
forward(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    _plan_fwd.execute_dft(&input[0], &output[0]);
}

template<class T>
T
isce::signal::Signal<T>::
inverse(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    _plan_inv.execute_dft(&input[0], &output[0]);
}


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

template<class T> 
T isce::signal::Signal<T>::
upsample(std::valarray<std::complex<T>> &signal,
         std::valarray<std::complex<T>> &signalUpsampled,
         int rows, int cols, int nfft, int upsampleFactor)
{
    // temporary storage for the spectrum
    std::valarray<std::complex<T>> spectrum(upsampleFactor*nfft*rows);

    // forward fft in range
    forwardRangeFFT(signal, spectrum,
                    cols, rows, 
                    upsampleFactor*nfft, rows);

    //shift the spectrum

    // inverse fft to get the upsampled signal
    inverseRangeFFT(spectrum, signalUpsampled,
                    upsampleFactor*nfft, rows, 
                    upsampleFactor*nfft, rows);
    
    // Normalize
    //signalUpsampled /=n fft;
}


// Declaration of the class
// We currently allow float and double. If at any time "long double" is needed, 
// declaration should be added here. 

template class isce::signal::Signal<float>;
template class isce::signal::Signal<double>;

//end of file
