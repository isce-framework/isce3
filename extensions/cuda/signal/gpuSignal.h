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
        gpuSignal(cufftType _type);
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

        /** moving data in between device and host */
        void dataToDevice(std::complex<T> *input);
        void dataToDevice(std::valarray<std::complex<T>> &input);
        void dataToHost(std::complex<T> *output);
        void dataToHost(std::valarray<std::complex<T>> &output);

        /** forward transforms without intermediate return */
        void forwardC2C();

        /** forward transforms with intermediate return */
        void forwardC2C(std::complex<T> *input, std::complex<T> *output);
        void forwardC2C(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void forwardZ2Z(std::complex<T> *input, std::complex<T> *output);
        void forwardZ2Z(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void forwardD2Z(T *input, std::complex<T> *output);

        /** inverse transforms using existing device memory **/
        void inverseC2C();

        /** inverse transforms */
        void inverseC2C(std::complex<T> *input, std::complex<T> *output);
        void inverseC2C(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void inverseZ2Z(std::complex<T> *input, std::complex<T> *output);
        void inverseZ2Z(std::valarray<std::complex<T>> &input,
                        std::valarray<std::complex<T>> &output);
        void inverseZ2D(std::complex<T> *input, T *output);

        int getRows() {return _rows;};
        int getColumns() {return _columns;};

    private:
        cufftHandle _plan;
        bool _plan_set;
        cufftType _cufft_type;

        // FFT plan parameters
        int _rank;
        int* _n;
        int _howmany;
        int* _inembed;
        int _istride;
        int _idist;
        int* _onembed;
        int _ostride;
        int _odist;
        int _n_elements;
        int _rows;
        int _columns;

        // device memory pointers
        T *_d_data;
        bool _d_data_set;
        T *_d_data_up;
        bool _d_data_up_set;
};

template<class T>
void shift(std::valarray<std::complex<T>> &input,
           std::valarray<std::complex<T>> &output,
           int rows, int nfft, int columns);

void upsampleC2C(std::valarray<std::complex<float>> &input,
                 std::valarray<std::complex<float>> &output,
                 std::valarray<std::complex<float>> &shiftImpact,
                 isce::cuda::signal::gpuSignal<float> &fwd,
                 isce::cuda::signal::gpuSignal<float> &inv);

void upsampleZ2Z(std::valarray<std::complex<double>> &input,
                 std::valarray<std::complex<double>> &output,
                 std::valarray<std::complex<double>> &shiftImpact,
                 isce::cuda::signal::gpuSignal<double> &fwd,
                 isce::cuda::signal::gpuSignal<double> &inv);

#endif

// end of file
