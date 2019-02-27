// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#ifndef ISCE_CUDA_SIGNAL_LOOKS_H
#define ISCE_CUDA_SIGNAL_LOOKS_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#define CUDA_GLOBAL
#endif

#include <valarray>
#include "isce/cuda/core/gpuComplex.h"

using isce::cuda::core::gpuComplex;

// Declaration
namespace isce {
    namespace cuda {
        namespace signal {
            template<class T>
            class gpuLooks;
        }
    }
}

// Definition
template<class T>
class isce::cuda::signal::gpuLooks {
    public:
        gpuLooks() {};
        ~gpuLooks() {};

        /** Multi-looking an array of real data */
        void multilook(std::valarray<T> &input,
                        std::valarray<T> &output);

        /** Multi-looking an array of real data */
        void multilook(std::valarray<T> &input,
                        std::valarray<T> &output,
                        std::valarray<T> &weights);

        /** Multi-looking an array of real data (excluding noData values) */     
        void multilook(std::valarray<T> &input,
                        std::valarray<T> &output,
                        T noDataValue);

        void multilook(std::valarray<std::complex<T>> &input,
                        std::valarray<T> &output, 
                        int p);

    private:
        // number of rows before multilooking/downsampling
        size_t _nrows;

        // number of columns before multilooking/downsampling
        size_t _ncols;

        // number of rows after multilooking/downsampling
        size_t _nrowsLooked;

        // number of columns after multilooking/downsampling
        size_t _ncolsLooked;

        // number of looks in range direction (columns) i.e. downsampling factor in range
        size_t _colsLooks;

        // numbe of looks in azimuth direction (rows) i.e. downsampling factor in azimuth
        size_t _rowsLooks;

};

template<class T>
CUDA_GLOBAL void multilooks_g(T *lo_res, T *hi_res, int rows_lo, int row_resize, int col_resize, int sz_lo);

template<class T>
CUDA_GLOBAL void multilooks_no_data_g(T *lo_res, T *hi_res, T no_data_value, T* weights, int rows_lo, int row_resize, int col_resize, int sz_lo);

template<class T>
CUDA_GLOBAL void multilooks_weighted_g(T *lo_res, T *hi_res, T* weights, int rows_lo, int row_resize, int col_resize, int sz_lo);

template<class T>
CUDA_GLOBAL void multilooks_power_g(T *lo_res, gpuComplex<T> *hi_res, int power, int rows_lo, int row_resize, int col_resize, int sz_lo);

#endif

// end of file
