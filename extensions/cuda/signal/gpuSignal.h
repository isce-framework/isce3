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
        gpuSignal(cufftType _type) : _cufft_type(_type) {};
        ~gpuSignal();

        /** \brief initiate plan for FFT in range direction 
         * for a block of complex data.
         * azimuth direction is assumed to be in the direction of the 
         * columns of the array.
         */
        void azimuthFFT(int ncolumns, int nrows);

        /** \brief initiate plan for FFT in azimuth direction 
         * for a block of complex data.
         * range direction is assumed to be in the direction of the 
         * columns of the array.
         */
        void rangeFFT(int ncolumns, int nrows);

        /** \brief initiate plan for FFT in azimuth direction 
         * for a block of complex data.
         * range direction is assumed to be in the direction of the 
         * columns of the array.
         */
        void FFT2D(int ncolumns, int nrows);

        /** \brief initiate cuFFT plan for a block of complex data
         *  input parameters cuFFT interface for fftw_plan_many_dft
         */
        void fftPlan(int rank, int* n, int howmany,                   
                    int* inembed, int istride, int idist,
                    int* onembed, int ostride, int odist);

        /** \brief next power of two*/
        void nextPowerOfTwo(size_t N, size_t& fftLength);

        /** \brief determine the required parameters for setting range FFT plans */
        inline void _configureRangeFFT(int ncolumns, int nrows);

        /** \brief determine the required parameters for setting azimuth FFT plans */
        inline void _configureAzimuthFFT(int ncolumns, int nrows);

        /** forward transforms */
        void forwardC2C(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void forwardZ2Z(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);

        /** inverse transforms */
        void inverseC2C(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void inverseZ2Z(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);

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
