// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#include "gpuSignal.h"

#include <cuda_runtime.h>
#include <cufftXt.h>
#include "isce/cuda/helper_cuda.h"
#include "isce/cuda/helper_functions.h"

using isce::cuda::signal::gpuSignal;


/** Destructor **/
template<class T>
gpuSignal<T>::
~gpuSignal() {
    cufftDestroy(_plan);
}

/**
*  @param[in] signal input block of data
*  @param[out] spectrum output block of spectrum
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void gpuSignal<T>::
forwardRangeFFT(int ncolumns, int nrows)
                
{

    _configureRangeFFT(ncolumns, nrows);
    
    fftPlanForward(_rank, _n, _howmany,
                    _inembed, _istride, _idist,
                    _onembed, _ostride, _odist);

}

/**
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
void gpuSignal<T>::
fftPlanForward(int rank, int *n, int howmany,
                int *inembed, int istride, int idist,
                int *onembed, int ostride, int odist)
{
    _cufft_type = CUFFT_C2C;
    checkCudaErrors(cufftCreate(&_plan));
    size_t worksize;
    checkCudaErrors(cufftMakePlanMany(_plan, rank, n, inembed,
                                      istride, idist, onembed, ostride, 
                                      odist, _cufft_type, _howmany, &worksize));

    //checkCudaErrors(cufftCreate(&_plan));checkCudaErrors(cufftMakePlan2d(_plan, n[0], n[1], _cufft_type, worksize));
}

template <class T>
void gpuSignal<T>::dbgTodos(int n, 
                           std::valarray<std::complex<T>> &input, 
                           std::valarray<std::complex<T>> &output)
{
    _cufft_type = CUFFT_C2C;
    checkCudaErrors(cufftPlan1d(&_plan, n, _cufft_type, 1));

    size_t input_size = input.size()*sizeof(std::complex<T>);
    size_t output_size = output.size()*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input from host to device
    checkCudaErrors(cudaMemcpy(d_input, &input[0], input_size, cudaMemcpyHostToDevice));

    // temp for input
    float *h_temp = reinterpret_cast<float *>(malloc(input_size));
    checkCudaErrors(cudaMemcpy(h_temp, d_input, input_size, cudaMemcpyDeviceToHost));
    std::fstream fin;
    fin.open("data_in.bin", std::ios::out | std::ios::binary);
    fin.write((char *)h_temp, input_size);
    fin.close();

    // transform in place
    checkCudaErrors(cufftExecC2C(_plan, reinterpret_cast<cufftComplex *>(d_input),
                                reinterpret_cast<cufftComplex *>(d_input),
                                CUFFT_FORWARD));

    // copy output from device to host
    checkCudaErrors(cudaMemcpy(&output[0], d_input, input_size, cudaMemcpyDeviceToHost));
    // redundant write out
    std::fstream fredun;
    fredun.open("data_out_redundant.bin", std::ios::out | std::ios::binary);
    fredun.write((char *)&output[0], input_size);
    fredun.close();
    
    // temp for output
    checkCudaErrors(cudaMemcpy(h_temp, d_input, input_size, cudaMemcpyDeviceToHost));
    std::fstream fout;
    fout.open("data_out.bin", std::ios::out | std::ios::binary);
    fout.write((char *)h_temp, input_size);
    fout.close();

    free(h_temp);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

/** @param[in] ncolumns number of columns
*   @param[in] nrows number of rows
*/
template <class T>
void gpuSignal<T>::
_configureRangeFFT(int ncolumns, int nrows)
{
    _rank = 1;                  // dimensionality of transform
    _n = new int[1];
    _n[0] = ncolumns;

    _howmany = nrows;
    
    _inembed = new int[1];
    _inembed[0] = ncolumns;

    _istride = 1;
    _idist = ncolumns;
    
    _onembed = new int[1];
    _onembed[0] = ncolumns;

    _ostride = 1;
    _odist = ncolumns;
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void isce::cuda::signal::gpuSignal<T>::
forward(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    size_t input_size = input.size()*sizeof(std::complex<T>);
    //size_t input_size = input.size()*sizeof(T)*2;
    size_t output_size = output.size()*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, &input[0], input_size, cudaMemcpyHostToDevice));

    // transform
    checkCudaErrors(cufftExecC2C(_plan, reinterpret_cast<cufftComplex *>(d_input),
                                reinterpret_cast<cufftComplex *>(d_input),
                                CUFFT_FORWARD));

    // copy output
    checkCudaErrors(cudaMemcpy(&output[0], d_input, input_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void isce::cuda::signal::gpuSignal<T>::
inverse(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    size_t input_size = input.size()*sizeof(std::complex<T>);
    //size_t input_size = input.size()*sizeof(T)*2;
    size_t output_size = output.size()*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, &input[0], input_size, cudaMemcpyHostToDevice));

    // transform
    checkCudaErrors(cufftExecC2C(_plan, reinterpret_cast<cufftComplex *>(d_input),
                                reinterpret_cast<cufftComplex *>(d_input),
                                CUFFT_INVERSE));

    // copy output
    checkCudaErrors(cudaMemcpy(&output[0], d_input, input_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

/**
*  @param[in] rank rank of the transform (1: for one dimensional and 2: for two dimensional transform)
*  @param[in] size size of each transform (ncols: for range FFT, nrows: for azimuth FFT)
*  @param[in] howmany number of FFT transforms for a block of data (nrows: for range FFT, ncols: for azimuth FFT)
*  @param[in] inembed
*  @param[in] istride
*  @param[in] idist
*  @param[in] onembed
*  @param[in] ostride
*  @param[in] odist
template <class T>
void
gpuSignal<T>::
fftPlanBackward(int rank, int *n, int howmany,
                int *inembed, int istride, int idist,
                int *onembed, int ostride, int odist)
{
    size_t worksize;
    checkCudaErrors(cufftCreate(&_plan_inv));
    checkCudaErrors(cufftMakePlanMany(_plan_inv, rank, n, inembed,
                                      istride, idist, onembed, ostride, 
                                      odist, _cufft_type, 1, worksize)
}

*/
/*
 each template parameter needs it's own declaration here
 */
template class gpuSignal<float>;
