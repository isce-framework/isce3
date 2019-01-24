// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#ifndef ISCE_CUDA_SIGNAL_SIGNAL_H
#define ISCE_CUDA_SIGNAL_SIGNAL_H

#include <complex>
#include <valarray>

#include <cufft.h>

// Declaration
namespace isce {
    namespace cuda {
        namespace signal {
            template<class T>
            class gpuSignal;
        }
    }
}

// Definition
template<class T>
class isce::cuda::signal::gpuSignal {

    public:
        // Default constructor
        gpuSignal() {};
        ~gpuSignal();

        /** \brief initiate plan for forward FFT in range direction 
         * for a block of complex data.
         * range direction is assumed to be in the direction of the 
         * columns of the array.
         */
        void forwardRangeFFT(int ncolumns, int nrows);

        /** \brief initiate forward FFTW3 plan for a block of complex data
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft
         */
        void fftPlanForward(int rank, int* n, int howmany,                   
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist);

        /** \brief determine the required parameters for setting range FFT plans */
        inline void _configureRangeFFT(int ncolumns, int nrows);

        /** forward transform */
        void forward(std::valarray<std::complex<T>> &input,
                    std::valarray<std::complex<T>> &output);

        /** \brief initiate forward FFTW3 plan for a block of complex data
         *  input parameters follow FFTW3 interface for fftw_plan_many_dft
        void fftPlanBackward(int rank, int* n, int howmany,                   
                            int* inembed, int istride, int idist,
                            int* onembed, int ostride, int odist);
         */

    private:
        cufftHandle _plan;
        cufftType _cufft_type;

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

#endif

// end of file
