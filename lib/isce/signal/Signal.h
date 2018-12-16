// -*- C++ -*-
// -*- coding: utf-8 -*-
// 
// Author: Heresh Fattahi
// Copyright 2018-
//

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

/** A class to handle 1D FFT in range and azimuth directions 
 *
 */
template<class T> 
class isce::signal::Signal {
    public:
        /** Default constructor. */ 
        Signal() {};

        /** Constructor with number of threads. This uses the Multi-threaded FFTW */
        Signal(int nthreads) {fftw3cxx::init_threads<T>(); fftw3cxx::plan_with_nthreads<T>(nthreads);};

        ~Signal() {};

        /** \brief initiate forward FFTW3 plan for a block of complex data
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft
         */
        void fftPlanForward(std::valarray<std::complex<T>> &input, 
                            std::valarray<std::complex<T>> &output,
                            int rank, int* n, int howmany,
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist, int sign);

        /** \brief initiate forward FFTW3 plan for a block of complex data
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft
         */
        void fftPlanForward(std::complex<T>* input,
                            std::complex<T>* output,
                            int rank, int* n, int howmany,
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist, int sign);

        /** \brief initiate forward FFTW3 plan for a block of real data
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft_r2c
         */
        void fftPlanForward(std::valarray<T> &input,
                            std::valarray<std::complex<T>> &output,
                            int rank, int* n, int howmany,
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist);

        /** \brief initiate forward FFTW3 plan for a block of real data
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft_r2c
         */
        void fftPlanForward(T* input,
                            std::complex<T>* output,
                            int rank, int* n, int howmany,
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist);

        /** \brief initiate iverse FFTW3 plan for a block of spectrum 
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft
         */
        void fftPlanBackward(std::valarray<std::complex<T>> &input,
                            std::valarray<std::complex<T>> &output,
                            int rank, int* n, int howmany,
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist, int sign);    

        /** \brief initiate iverse FFTW3 plan for a block of spectrum
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft
         */
        void fftPlanBackward(std::complex<T>* input,
                            std::complex<T>* output,
                            int rank, int* n, int howmany,
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist, int sign);

        /** \brief initiate iverse FFTW3 plan for a block of spectrum
         * to be transformed to real data
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft_c2r
         */
        void fftPlanBackward(std::valarray<std::complex<T>>& input,
                            std::valarray<T>& output,
                            int rank, int* n, int howmany,
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist);

        /** \brief initiate iverse FFTW3 plan for a block of spectrum
         * to be transformed to real data
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft_c2r
         */
        void fftPlanBackward(std::complex<T>* input,
                            T* output,
                            int rank, int* n, int howmany,
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist);

        /** forward transform */
        void forward(std::valarray<std::complex<T>> &input,
                    std::valarray<std::complex<T>> &output);

        /** forward transform */
        void forward(std::complex<T>* input,
                    std::complex<T>* output);

        /** forward transform */
        void forward(std::valarray<T> &input,
                    std::valarray<std::complex<T>> &output);

        /** forward transform */
        void forward(T* input,
                    std::complex<T>* output);

        

        /** inverse transform*/
        void inverse(std::valarray<std::complex<T>> &input,
                    std::valarray<std::complex<T>> &output);

        /** inverse transform*/
        void inverse(std::complex<T>* input,
                    std::complex<T>* output);

        /** inverse transform*/
        void inverse(std::valarray<std::complex<T>>& input,
                    std::valarray<T>& output);

        /** inverse transform*/
        void inverse(std::complex<T>* input,
                    T* output);

        /** \brief initiate plan for forward FFT in range direction 
         * for a block of complex data.
         * range direction is assumed to be in the direction of the 
         * columns of the array.
         */
        void forwardRangeFFT(std::valarray<std::complex<T>>& signal, 
                    std::valarray<std::complex<T>>& spectrum,
                    int ncolumns, int nrows);

        /** \brief initiate plan for forward FFT in range direction
         * for a block of complex data.
         * range direction is assumed to be in the direction of the
         * columns of the array.
         */
        void forwardRangeFFT(std::complex<T>* signal,
                    std::complex<T>* spectrum,
                    int ncolumns, int nrows);

        /** \brief initiate plan for forward FFT in range direction
         * for a block of real data.
         * range direction is assumed to be in the direction of the
         * columns of the array.
         */
        void forwardRangeFFT(std::valarray<T>& signal,
                    std::valarray<std::complex<T>>& spectrum,
                    int ncolumns, int nrows);

        /** \brief initiate plan for forward FFT in range direction
         * for a block of real data.
         * range direction is assumed to be in the direction of the
         * columns of the array.
         */
        void forwardRangeFFT(T* signal,
                    std::complex<T>* spectrum,
                    int ncolumns, int nrows);

        /** \brief initiate plan for forward FFT in azimuth direction 
         * for a block of complex data.
         * azimuth direction is assumed to be in the direction of the
         * rows of the array.
         */
        void forwardAzimuthFFT(std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                int ncolumns, int nrows);

        /** \brief initiate plan for forward FFT in azimuth direction
         * for a block of complex data.
         * azimuth direction is assumed to be in the direction of the
         * rows of the array.
         */
        void forwardAzimuthFFT(std::complex<T>* signal,
                                std::complex<T>* spectrum,
                                int ncolumns, int nrows);

        /** \brief initiate plan for forward FFT in azimuth direction
         * for a block of real data.
         * azimuth direction is assumed to be in the direction of the
         * rows of the array.
         */
        void forwardAzimuthFFT(std::valarray<T> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                int ncolumns, int nrows);

        /** \brief initiate plan for forward FFT in azimuth direction
         * for a block of real data.
         * azimuth direction is assumed to be in the direction of the
         * rows of the array.
         */
        void forwardAzimuthFFT(T* signal,
                                std::complex<T>* spectrum,
                                int ncolumns, int nrows);

        /** \brief initiate plan for forward two imensional FFT for a block of complex data
         */
        void forward2DFFT(std::valarray<std::complex<T>> &signal,
                                std::valarray<std::complex<T>> &spectrum,
                                int ncolumns, int nrows);
        
        /** \brief initiate plan for forward two imensional FFT for a block of complex data
         */
        void forward2DFFT(std::complex<T>* signal,
                                std::complex<T>* spectrum,
                                int ncolumns, int nrows);

        /** \brief initiate plan for forward two imensional FFT for a block of real data
         */
        void forward2DFFT(std::valarray<T>& signal,
                            std::valarray<std::complex<T>>& spectrum,
                            int ncolumns, int nrows);

        /** \brief initiate plan for forward two imensional FFT for a block of real data
         */
        void forward2DFFT(T* signal,
                            std::complex<T>* spectrum,
                            int ncolumns, int nrows);

        /** \brief initiate plan for backward FFT in range direction for a block of data
         */
        void inverseRangeFFT(std::valarray<std::complex<T>> &spectrum, 
                            std::valarray<std::complex<T>> &signal,
                            int ncolumns, int nrows);

        /** \brief initiate plan for backward FFT in range direction for a block of data
         */
        void inverseRangeFFT(std::complex<T>* spectrum,
                            std::complex<T>* signal,
                            int ncolumns, int nrows);

        /** \brief initiate plan for backward FFT in range direction for a block of data
         */
        void inverseRangeFFT(std::valarray<std::complex<T>> &spectrum,
                            std::valarray<T> &signal,
                            int ncolumns, int nrows);

        /** \brief initiate plan for backward FFT in range direction for a block of data
         */
        void inverseRangeFFT(std::complex<T>* spectrum,
                            T* signal,
                            int ncolumns, int nrows);

        /** \brief initiate plan for inverse FFT in azimuth direction for a block of data
         */
        void inverseAzimuthFFT(std::valarray<std::complex<T>> &spectrum,
                                std::valarray<std::complex<T>> &signal,
                                int ncolumns, int nrows);

        /** \brief initiate plan for inverse FFT in azimuth direction for a block of data
        */
        void inverseAzimuthFFT(std::complex<T>* spectrum,
                                std::complex<T>* signal,
                                int ncolumns, int nrows);
        /** \brief initiate plan for inverse FFT in azimuth direction for a block of data
         */
        void inverseAzimuthFFT(std::valarray<std::complex<T>> &spectrum,
                                std::valarray<T> &signal,
                                int ncolumns, int nrows);

        /** \brief initiate plan for inverse FFT in azimuth direction for a block of data
         */
        void inverseAzimuthFFT(std::complex<T>* spectrum,
                                T* signal,
                                int ncolumns, int nrows);

        /** \brief initiate plan for inverse two dimensional FFT for a block of data
         */
        void inverse2DFFT(std::valarray<std::complex<T>> &spectrum,
                            std::valarray<std::complex<T>> &signal,
                            int ncolumns, int nrows);

        /** \brief initiate plan for inverse two dimensional FFT for a block of data
         */
        void inverse2DFFT(std::complex<T>* spectrum,
                            std::complex<T>* signal,
                            int ncolumns, int nrows);
        
        /** \brief initiate plan for inverse two dimensional FFT for a block of data
         */
        void inverse2DFFT(std::valarray<std::complex<T>> &spectrum,
                            std::valarray<T> &signal,
                            int ncolumns, int nrows);

        /** \brief initiate plan for inverse two dimensional FFT for a block of data
         */
        void inverse2DFFT(std::complex<T>* spectrum,
                            T* signal,
                            int ncolumns, int nrows);


        /** \brief upsampling a block of data in range direction */
        void upsample(std::valarray<std::complex<T>> &signal,
                    std::valarray<std::complex<T>> &signalOversampled,
                    int rows, int nfft, int oversampleFactor);

        /** \brief upsampling a block of data in range direction and shifting 
         * the upsampled signal by a constant. The shift is applied by an 
         * inout linear phase term in frequency domain. 
         */
        void upsample(std::valarray<std::complex<T>> &signal,
                    std::valarray<std::complex<T>> &signalOversampled,
                    int rows, int nfft, int oversampleFactor, 
                    std::valarray<std::complex<T>> shiftImpact);

        /** \brief next power of two*/
        inline void nextPowerOfTwo(size_t N, size_t& fftLength);

        /** \brief determine the required parameters for setting range FFT plans */
        inline void _configureRangeFFT(int ncolumns, int nrows);

        /** \brief determine the required parameters for setting azimuth FFT plans */
        inline void _configureAzimuthFFT(int ncolumns, int nrows);

        /** \brief determine the required parameters for setting 2D FFT plans */
        inline void _configure2DFFT(int ncolumns, int nrows);

    private:
        isce::fftw3cxx::plan<T> _plan_fwd;
        isce::fftw3cxx::plan<T> _plan_inv;

        int _rank;
        int* _n;
        int _howmany;
        int* _inembed;
        int _istride;
        int _idist;
        int* _onembed;
        int _ostride;
        int _odist;

};

#define ISCE_SIGNAL_SIGNAL_ICC
#include "Signal.icc"
#undef ISCE_SIGNAL_SIGNAL_ICC

#endif


