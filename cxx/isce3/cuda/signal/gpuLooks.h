#pragma once

#include "forward.h"

#include <valarray>
#include <thrust/complex.h>
#include <isce3/core/Common.h>

namespace isce3::cuda::signal {

template<class T>
class gpuLooks {
    public:
        gpuLooks() {};
        ~gpuLooks() {};

        /** Multi-looking an array of real data */
        void multilook(std::valarray<T> &input,
                       std::valarray<T> &output);

        void multilook(std::valarray<std::complex<T>> &input,
                       std::valarray<std::complex<T>> &output);

        /** Multi-looking an array of real data */
        void multilook(std::valarray<T> &input,
                       std::valarray<T> &output,
                       std::valarray<T> &weights);

        /** Multi-looking an array of real data (excluding noData values) */
        void multilook(std::valarray<T> &input,
                       std::valarray<T> &output,
                       T noDataValue);
        void multilook(std::valarray<std::complex<T>> &input,
                       std::valarray<std::complex<T>> &output,
                       std::complex<T> noDataValue);

        /** POWER! */
        void multilook(std::valarray<std::complex<T>> &input,
                       std::valarray<T> &output,
                       int p);

        void nrows(size_t n) { _nrows = n; };
        void ncols(size_t n) { _ncols = n; };
        void nrowsLooked(size_t n) { _nrowsLooked = n; };
        void ncolsLooked(size_t n) { _ncolsLooked = n; };
        void rowsLooks(size_t n) { _rowsLooks = n; };
        void colsLooks(size_t n) { _colsLooks = n; };
        size_t nrows() { return _nrows; };
        size_t ncols() { return _ncols; };
        size_t nrowsLooked() { return _nrowsLooked; };
        size_t ncolsLooked() { return _ncolsLooked; };
        size_t rowsLooks() { return _rowsLooks; };
        size_t colsLooks() { return _colsLooks; };

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
CUDA_GLOBAL void multilooks_g(T *lo_res,
        const T* __restrict__ hi_res,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz);

template<class T>
CUDA_GLOBAL void multilooks_g(thrust::complex<T> *lo_res,
        const thrust::complex<T>* __restrict__ hi_res,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz);

template<class T>
CUDA_GLOBAL void multilooks_no_data_g(T *lo_res,
        const T* __restrict__ hi_res,
        T no_data_value,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz);

template<class T>
CUDA_GLOBAL void multilooks_no_data_g(thrust::complex<T> *lo_res,
        const thrust::complex<T>* __restrict__ hi_res,
        thrust::complex<T> no_data_value,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz);

template<class T>
CUDA_GLOBAL void multilooks_weighted_g(T *lo_res,
        const T* __restrict__ hi_res,
        const T* __restrict__ weights,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo);

template<class T>
CUDA_GLOBAL void multilooks_power_g(T *lo_res,
        const thrust::complex<T>* __restrict__ hi_res,
        int power,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz);

} // namespace isce3::cuda::signal
