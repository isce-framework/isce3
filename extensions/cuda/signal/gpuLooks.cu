// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#include "gpuLooks.h"
#include "isce/cuda/helper_cuda.h"
#include "isce/cuda/helper_functions.h"

#define THRD_PER_BLOCK 1024 // Number of threads per block (should always %32==0)

template <typename T>
void isce::cuda::signal::gpuLooks<T>::multilook(std::valarray<T> &lo_res, 
                                                std::valarray<T> &hi_res,
                                                T noDataValue) 
{
    // allocate lo res output on device
    T *d_lo_res;
    size_t lo_res_size = _nrowsLooked*_ncolsLooked*sizeof(T)*2;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lo_res), lo_res_size));

    // allocate and copy to device hi res input
    T *d_hi_res;
    size_t hi_res_size = _nrows*_ncols*sizeof(T)*2;
    // allocate input
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_hi_res), hi_res_size));
    // copy hi_res
    checkCudaErrors(cudaMemcpy(d_hi_res, &hi_res[0], hi_res_size, cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((lo_res_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // run kernels
    multlooks_no_data_g<<<grid, block>>>(d_lo_res, d_hi_res, noDataValue, _nrowsLooked, _rowsLooks, _colsLooks, _nrowsLooked*_ncolsLooked);

    // copy from device lo res output
    checkCudaErrors(cudaMemcpy(&lo_res[0], d_lo_res, lo_res_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_lo_res));
    checkCudaErrors(cudaFree(d_hi_res));
}

template <typename T>
void isce::cuda::signal::gpuLooks<T>::multilook(std::valarray<T> &lo_res, 
                                                std::valarray<T> &hi_res) 
{
    // allocate lo res output on device
    T *d_lo_res;
    size_t lo_res_size = _nrowsLooked*_ncolsLooked*sizeof(T)*2;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lo_res), lo_res_size));

    // allocate and copy to device hi res input
    T *d_hi_res;
    size_t hi_res_size = _nrows*_ncols*sizeof(T)*2;
    // allocate input
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_hi_res), hi_res_size));
    // copy hi_res
    checkCudaErrors(cudaMemcpy(d_hi_res, &hi_res[0], hi_res_size, cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((lo_res_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // run kernels
    multlooks_g<<<grid, block>>>(d_lo_res, d_hi_res, _nrowsLooked, _rowsLooks, _colsLooks, _nrowsLooked*_ncolsLooked);

    // copy from device lo res output
    checkCudaErrors(cudaMemcpy(&lo_res[0], d_lo_res, lo_res_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_lo_res));
    checkCudaErrors(cudaFree(d_hi_res));
}


template <typename T>
void isce::cuda::signal::gpuLooks<T>::multilook(std::valarray<T> &lo_res, 
                                                std::valarray<T> &hi_res, 
                                                std::valarray<T> &weights) 
{
    // allocate lo res output on device
    T *d_lo_res;
    size_t lo_res_size = _nrowsLooked*_ncolsLooked*sizeof(T)*2;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lo_res), lo_res_size));

    // allocate and copy to device hi res input
    T *d_hi_res;
    size_t hi_res_size = _nrows*_ncols*sizeof(T)*2;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_hi_res), hi_res_size));
    checkCudaErrors(cudaMemcpy(d_hi_res, &hi_res[0], hi_res_size, cudaMemcpyHostToDevice));

    // allocate and copy to device weights
    T *d_weights;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_weights), hi_res_size));
    checkCudaErrors(cudaMemcpy(d_weights, &weights[0], hi_res_size, cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((lo_res_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // run kernels
    multlooks_weighted_g<<<grid, block>>>(d_lo_res, d_hi_res, d_weights, _nrowsLooked, _rowsLooks, _colsLooks, _nrowsLooked*_ncolsLooked);

    // copy from device lo res output
    checkCudaErrors(cudaMemcpy(&lo_res[0], d_lo_res, lo_res_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_lo_res));
    checkCudaErrors(cudaFree(d_hi_res));
    checkCudaErrors(cudaFree(d_weights));
}


/*
   accumulate then average from hi res to lo res
output:
   lo_res
input:
   hi_res to be reduced to lo_res
   rows_lo rows in lo res
   row_resize scale factor of hi to lo in rows
   col_resize scale factor of hi to lo in cols
   sz_lo number of elements in lo res
 */
template <typename T>
__global__ void multilooks_g(T *lo_res, T *hi_res, int rows_lo, int row_resize, int col_resize, int sz_lo) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < sz_lo) {
        int i_lo_row = i % rows_lo;
        int i_lo_col = i % rows_lo;
        int blk_sz = row_resize * col_resize;

        // loop over contributing lo_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            int i_hi_row = i_blk_row + i_lo_row*row_resize;
            // loop over contributing lo_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                int i_hi_col = i_blk_col + i_lo_col*col_resize;
                // combine lo_res row and col index to hi_res index
                int i_hi = i_hi_row*rows_lo*row_resize + i_hi_col;
                // accumulate lo_res into lo_res
                lo_res[i] += hi_res[i_hi];
            }
        }
        lo_res[i] /= blk_sz;
    }
}


/*
   accumulate then average from hi res to lo res
output:
   lo_res
input:
   hi_res to be reduced to lo_res
   no_data_value values where hi_res data is not accounted for
   rows_lo rows in lo res
   row_resize scale factor of hi to lo in rows
   col_resize scale factor of hi to lo in cols
   sz_lo number of elements in lo res
 */
template <typename T>
__global__ void multilooks_no_data_g(T *lo_res, T *hi_res, T no_data_value, int rows_lo, int row_resize, int col_resize, int sz_lo) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < sz_lo) {
        int i_lo_row = i % rows_lo;
        int i_lo_col = i % rows_lo;
        int blk_sz = row_resize * col_resize;

        // 
        int n_no_val = 0;

        // loop over contributing lo_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            int i_hi_row = i_blk_row + i_lo_row*row_resize;
            // loop over contributing lo_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                int i_hi_col = i_blk_col + i_lo_col*col_resize;
                // combine lo_res row and col index to hi_res index
                int i_hi = i_hi_row*rows_lo*row_resize + i_hi_col;
                // accumulate lo_res into lo_res
                T hi_res_pixel_value = hi_res[i_hi];
                if (hi_res_pixel_value != no_data_value)
                    lo_res[i] += hi_res_pixel_value;
                else
                    ++n_no_val;
            }
        }
        lo_res[i] /= (blk_sz - n_no_val);
    }
}


/*
   accumulate, apply weight, then average from hi res to lo res
output:
   lo_res
input:
   hi_res to be reduced to lo_res
   weights weights to be applied to lo_res data
   rows_lo rows in lo res
   row_resize scale factor of hi to lo in rows
   col_resize scale factor of hi to lo in cols
   sz_lo number of elements in lo res
 */
template <typename T>
__global__ void multilooks_weighted_g(T *lo_res, T *hi_res, T* weights, int rows_lo, int row_resize, int col_resize, int sz_lo) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < sz_lo) {
        int i_lo_row = i % rows_lo;
        int i_lo_col = i % rows_lo;
        int blk_sz = row_resize * col_resize;

        T sum_weight = 0;
        // loop over contributing hi_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            int i_hi_row = i_blk_row + i_lo_row*row_resize;
            // loop over contributing lo_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                int i_hi_col = i_blk_col + i_lo_col*col_resize;
                // combine lo_res row and col index to hi_res index
                int i_hi = i_hi_row*rows_lo*row_resize + i_hi_col;
                // accumulate lo_res into lo_res
                lo_res[i] += hi_res[i_hi];
                sum_weight += weights[i_hi];
            }
        }

        if (sum_weight > 0) {
            lo_res[i] /= weights[i];
        } else {
            lo_res[i] == 0;
        }
    }
}

/*
   accumulate then average from hi res to lo res
output:
   lo_res
input:
   hi_res to be reduced to lo_res
   power 
   rows_lo rows in lo res
   row_resize scale factor of hi to lo in rows
   col_resize scale factor of hi to lo in cols
   sz_lo number of elements in lo res
 */
template <typename T>
__global__ void multilooks_power_g(T *lo_res, gpuComplex<T> *hi_res, int power, int rows_lo, int row_resize, int col_resize, int sz_lo) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < sz_lo) {
        int i_lo_row = i % rows_lo;
        int i_lo_col = i % rows_lo;
        int blk_sz = row_resize * col_resize;

        // loop over contributing lo_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            int i_hi_row = i_blk_row + i_lo_row*row_resize;
            // loop over contributing lo_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                int i_hi_col = i_blk_col + i_lo_col*col_resize;
                // combine lo_res row and col index to hi_res index
                int i_hi = i_hi_row*rows_lo*row_resize + i_hi_col;
                // accumulate lo_res into lo_res
                lo_res[i] += pow(abs(hi_res[i_hi]), power);
            }
        }
        lo_res[i] /= blk_sz;
    }
}
