// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Liang Yu
// Copyright 2019

#include <typeinfo>
#include <cstdio>
#include <string>

#include "gpuSignal.h"

#include <cuda_runtime.h>
#include <cufftXt.h>
#include "isce/cuda/helper_cuda.h"
#include "isce/cuda/helper_functions.h"

using isce::cuda::signal::gpuSignal;


/** Constructor **/
template<class T>
gpuSignal<T>::
gpuSignal(cufftType _type) {
    _cufft_type = _type;
    _plan_set = false;
    _d_data_set = false;

    _n = new int[2];
    _inembed = new int[2];
    _onembed = new int[2];
}

/** Destructor **/
template<class T>
gpuSignal<T>::
~gpuSignal() {
    if (_plan_set)
        cufftDestroy(_plan);

    if (_d_data_set)
        cudaFree(_d_data);

    delete[] _n;
    delete[] _inembed;
    delete[] _onembed;
}

/**
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void gpuSignal<T>::
rangeFFT(int ncolumns, int nrows)
{
    _configureRangeFFT(ncolumns, nrows);
    
    fftPlan(_rank, _n, _howmany,
            _inembed, _istride, _idist,
            _onembed, _ostride, _odist);
}

/**
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void gpuSignal<T>::
azimuthFFT(int ncolumns, int nrows)
{
    _configureAzimuthFFT(ncolumns, nrows);
    
    fftPlan(_rank, _n, _howmany,
            _inembed, _istride, _idist,
            _onembed, _ostride, _odist);
}

/**
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void gpuSignal<T>::
FFT2D(int ncolumns, int nrows)
{
    _n_elements = nrows * ncolumns;
    if (_plan_set)
        cufftDestroy(_plan);

    checkCudaErrors(cufftCreate(&_plan));
    size_t worksize;
    checkCudaErrors(cufftMakePlan2d(_plan, nrows, ncolumns, _cufft_type, &worksize));
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
fftPlan(int rank, int *n, int howmany,
        int *inembed, int istride, int idist,
        int *onembed, int ostride, int odist)
{
    if (_plan_set)
        cufftDestroy(_plan);

    checkCudaErrors(cufftCreate(&_plan));
    _plan_set = true;
    size_t worksize;
    checkCudaErrors(cufftMakePlanMany(_plan, rank, n, 
                                      inembed, istride, idist, 
                                      onembed, ostride, odist, 
                                      _cufft_type, _howmany, &worksize));
}

/** @param[in] N the actual length of a signal
*   @param[in] fftLength next power of two 
*/
template <class T>
void gpuSignal<T>::
nextPowerOfTwo(size_t N, size_t &fftLength)
{
    for (size_t i = 0; i < 17; ++i) {
        fftLength = std::pow(2, i);
        if (fftLength >= N) {
            break;
        }
    }
}

/** @param[in] ncolumns number of columns
*   @param[in] nrows number of rows
*/
template <class T>
void gpuSignal<T>::
_configureRangeFFT(int ncolumns, int nrows)
{
    _rank = 1;
    _n[0] = ncolumns;

    _howmany = nrows;
    
    _inembed[0] = ncolumns;

    _istride = 1;
    _idist = ncolumns;
    
    _onembed[0] = ncolumns;

    _ostride = 1;
    _odist = ncolumns;

    _n_elements = nrows * ncolumns;

    _rows = nrows;
    _columns = ncolumns;
}

/** @param[in] ncolumns number of columns
*   @param[in] nrows number of rows
*/
template <class T>
void gpuSignal<T>::
_configureAzimuthFFT(int ncolumns, int nrows)
{
    _rank = 1;
    _n[0] = nrows;

    _howmany = ncolumns;

    _inembed[0] = nrows;

    _istride = ncolumns;
    _idist = 1;

    _onembed[0] = nrows;

    _ostride = ncolumns;
    _odist = 1;

    _n_elements = nrows * ncolumns;

    _rows = nrows;
    _columns = ncolumns;
}

template<class T>
void gpuSignal<T>::
dataToDevice(std::complex<T> *input)
{
    if (!_d_data_set) {
        size_t input_size = _n_elements*sizeof(T)*2;
        // allocate input
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_data), input_size));
        // copy input
        checkCudaErrors(cudaMemcpy(_d_data, input, input_size, cudaMemcpyHostToDevice));
        _d_data_set = true;
    }
}

template<class T>
void gpuSignal<T>::
dataToDevice(std::valarray<std::complex<T>> &input)
{
    if (!_d_data_set) {
        size_t input_size = input.size()*sizeof(T)*2;
        // allocate input
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_data), input_size));
        // copy input
        checkCudaErrors(cudaMemcpy(_d_data, &input[0], input_size, cudaMemcpyHostToDevice));
        _d_data_set = true;
    }
}

template<class T>
void gpuSignal<T>::
dataToHost(std::complex<T> *output)
{
    if (_d_data_set) {
        size_t output_size = _n_elements*sizeof(T)*2;
        // copy output 
        checkCudaErrors(cudaMemcpy(output, _d_data, output_size, cudaMemcpyDeviceToHost));
    }
}

template<class T>
void gpuSignal<T>::
dataToHost(std::valarray<std::complex<T>> &output)
{
    if (_d_data_set) {
        size_t output_size = _n_elements*sizeof(T)*2;
        // copy output 
        checkCudaErrors(cudaMemcpy(&output[0], _d_data, output_size, cudaMemcpyDeviceToHost));
    }
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void gpuSignal<T>::
forwardC2C()
{
    if (_plan_set && _d_data_set)
        // transform
        checkCudaErrors(cufftExecC2C(_plan, reinterpret_cast<cufftComplex *>(_d_data),
                                    reinterpret_cast<cufftComplex *>(_d_data),
                                    CUFFT_FORWARD));
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void gpuSignal<T>::
forwardC2C(std::complex<T> *input, std::complex<T> *output)
{
    size_t input_size = _n_elements*sizeof(T)*2;
    size_t output_size = _n_elements*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));

    // transform
    checkCudaErrors(cufftExecC2C(_plan, reinterpret_cast<cufftComplex *>(d_input),
                                reinterpret_cast<cufftComplex *>(d_input),
                                CUFFT_FORWARD));

    // copy output
    checkCudaErrors(cudaMemcpy(output, d_input, input_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void gpuSignal<T>::
forwardC2C(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    size_t input_size = input.size()*sizeof(T)*2;
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
void gpuSignal<T>::
forwardZ2Z(std::complex<T> *input, std::complex<T> *output)
{
    size_t input_size = _n_elements*sizeof(T)*2;
    size_t output_size = _n_elements*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));

    // transform
    checkCudaErrors(cufftExecZ2Z(_plan, reinterpret_cast<cufftDoubleComplex *>(d_input),
                                reinterpret_cast<cufftDoubleComplex *>(d_input),
                                CUFFT_FORWARD));

    // copy output
    checkCudaErrors(cudaMemcpy(output, d_input, input_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

template<class T>
void gpuSignal<T>::
forwardZ2Z(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    size_t input_size = input.size()*sizeof(T)*2;
    size_t output_size = output.size()*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, &input[0], input_size, cudaMemcpyHostToDevice));

    // transform
    checkCudaErrors(cufftExecZ2Z(_plan, reinterpret_cast<cufftDoubleComplex *>(d_input),
                                reinterpret_cast<cufftDoubleComplex *>(d_input),
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
void gpuSignal<T>::
forwardD2Z(T *input, std::complex<T> *output)
{
    size_t input_size = _n_elements*sizeof(T);
    size_t output_size = _n_elements*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));

    // transform (implicitly forward)
    checkCudaErrors(cufftExecD2Z(_plan, reinterpret_cast<cufftDoubleReal *>(d_input),
                                reinterpret_cast<cufftDoubleComplex *>(d_output)));

    // copy output
    checkCudaErrors(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void gpuSignal<T>::
inverseC2C()
{
    if (_plan_set && _d_data_set)
        // transform
        checkCudaErrors(cufftExecC2C(_plan, reinterpret_cast<cufftComplex *>(_d_data),
                                    reinterpret_cast<cufftComplex *>(_d_data),
                                    CUFFT_INVERSE));
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void gpuSignal<T>::
inverseC2C(std::complex<T> *input, std::complex<T> *output)
{
    size_t input_size = _n_elements*sizeof(T)*2;
    size_t output_size = _n_elements*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));

    // transform
    checkCudaErrors(cufftExecC2C(_plan, reinterpret_cast<cufftComplex *>(d_input),
                                reinterpret_cast<cufftComplex *>(d_input),
                                CUFFT_INVERSE));

    // copy output
    checkCudaErrors(cudaMemcpy(output, d_input, input_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void gpuSignal<T>::
inverseC2C(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    size_t input_size = input.size()*sizeof(T)*2;
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

/** unnormalized forward transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void gpuSignal<T>::
inverseZ2Z(std::complex<T> *input, std::complex<T> *output)
{
    size_t input_size = _n_elements*sizeof(T)*2;
    size_t output_size = _n_elements*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));

    // transform
    checkCudaErrors(cufftExecZ2Z(_plan, reinterpret_cast<cufftDoubleComplex *>(d_input),
                                reinterpret_cast<cufftDoubleComplex *>(d_input),
                                CUFFT_INVERSE));

    // copy output
    checkCudaErrors(cudaMemcpy(output, d_input, input_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

template<class T>
void gpuSignal<T>::
inverseZ2Z(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    size_t input_size = input.size()*sizeof(T)*2;
    size_t output_size = output.size()*sizeof(T)*2;

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, &input[0], input_size, cudaMemcpyHostToDevice));

    // transform
    checkCudaErrors(cufftExecZ2Z(_plan, reinterpret_cast<cufftDoubleComplex *>(d_input),
                                reinterpret_cast<cufftDoubleComplex *>(d_input),
                                CUFFT_INVERSE));

    // copy output
    checkCudaErrors(cudaMemcpy(&output[0], d_input, input_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

/** unnormalized inverse transform
*  @param[in] input block of data
*  @param[out] output block of spectrum
*/
template<class T>
void gpuSignal<T>::
inverseZ2D(std::complex<T> *input, T *output)
{
    size_t input_size = _n_elements*sizeof(T)*2;
    size_t output_size = _n_elements*sizeof(T);

    // allocate device memory 
    T *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    T *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

    // copy input
    checkCudaErrors(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));

    // transform (implicitly inverse)
    checkCudaErrors(cufftExecZ2D(_plan, reinterpret_cast<cufftDoubleComplex *>(d_input),
                                reinterpret_cast<cufftDoubleReal *>(d_output)));

    // copy output
    checkCudaErrors(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

/** 1D shift (in range only)
TODO move to GPU to eliminate
*  lo res N x M_lo copied into hi res N x M_hi
*  where M_hi = f_upsample x M_lo
*  @param[in]
*  @param[out]
*  @param[in]
*  @param[in]
*  @param[in]
*/
template<class T>
void shift(std::valarray<std::complex<T>> &spectrum,
           std::valarray<std::complex<T>> &spectrumShifted,
           int rows, int nfft, int columns)
{
    //spectrum /=nfft;
    //shift the spectrum
    // The spectrum has values from begining to nfft index for each line. We want
    // to put the spectrum in correct ouput locations such that the spectrum of
    // the upsampled data has values from 0 to nfft/2 and from upsampleFactor*nfft - nfft/2 to the end.
    // For a 1D example:
    //      spectrum = [1,2,3,4,5,6,0,0,0,0,0,0]
    //  becomes:
    //      spectrumShifted = [1,2,3,0,0,0,0,0,0,4,5,6]
    size_t right_offset = columns - nfft/2;
    for (size_t i_row = 0; i_row < rows; ++i_row) {
        size_t row_offset_lo = i_row * nfft;
        size_t row_offset_hi = i_row * columns;
        // copy left side of lo res to left side of hi res
        spectrumShifted[std::slice(row_offset_hi, nfft/2, 1)] = spectrum[std::slice(row_offset_lo, nfft/2, 1)];
        // copy right side of lo res to right side of hi res
        spectrumShifted[std::slice(row_offset_hi + right_offset, nfft/2, 1)] = spectrum[std::slice(row_offset_lo+nfft/2, nfft/2, 1)];
    }
}

/**
    recast inputs to either cufftComplex or cufftDoubleComplex
*/
template<class T>
__global__ void rangeShift(T *data_lo_res, T *data_hi_res, int n_rows, int n_cols_lo, int n_cols_hi)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int i_row = i / n_rows;
    int i_col = i % n_cols_lo;

    if (i < n_cols_lo * n_rows) {
        if (i_col < n_cols_lo / 2)
            data_hi_res[i*n_cols_hi + i_col] = data_lo_res[i]; 
        else
            data_hi_res[(i+1) * n_cols_hi - (n_cols_lo-i_col)] = data_lo_res[i];
    }
}

void upsampleC2C(std::valarray<std::complex<float>> &input,
                 std::valarray<std::complex<float>> &output,
                 std::valarray<std::complex<float>> &shiftImpact, 
                 isce::cuda::signal::gpuSignal<float> &fwd,
                 isce::cuda::signal::gpuSignal<float> &inv)
{
    // temporary storage for the spectrum before and after the shift
    std::valarray<std::complex<float>> spectrum(input.size());
    std::valarray<std::complex<float>> spectrumShifted(output.size());

    spectrumShifted = std::complex<float> (0.0,0.0);

    // transform
    fwd.forwardC2C(input, spectrum);

    // shift data prior to upsampling transform
    shift<float>(spectrum, spectrumShifted, fwd.getRows(), fwd.getColumns(), inv.getColumns());

    // transform with upsampled spectrum
    inv.inverseC2C(spectrumShifted, output);

    // multiply the shiftImpact (a linear phase is frequency domain
    // equivalent to a shift in time domain) by the spectrum
    if (spectrumShifted.size() == shiftImpact.size())
        spectrumShifted *= shiftImpact;
}

void upsampleZ2Z(std::valarray<std::complex<double>> &input,
                 std::valarray<std::complex<double>> &output,
                 std::valarray<std::complex<double>> &shiftImpact, 
                 isce::cuda::signal::gpuSignal<double> &fwd,
                 isce::cuda::signal::gpuSignal<double> &inv)
{
    // temporary storage for the spectrum before and after the shift
    std::valarray<std::complex<double>> spectrum(input.size());
    std::valarray<std::complex<double>> spectrumShifted(output.size());

    spectrum = std::complex<double> (0.0,0.0);
    spectrumShifted = std::complex<double> (0.0,0.0);

    // transform
    fwd.forwardZ2Z(input, spectrum);

    // shift data prior to upsampling transform
    shift<double>(spectrum, spectrumShifted, fwd.getRows(), fwd.getColumns(), inv.getColumns());

    // transform
    inv.inverseZ2Z(spectrumShifted, output);

    // multiply the shiftImpact (a linear phase is frequency domain
    // equivalent to a shift in time domain) by the spectrum
    if (spectrumShifted.size() == shiftImpact.size())
        spectrumShifted *= shiftImpact;
}

/*
 each template parameter needs it's own declaration here
 */
template class gpuSignal<float>;
template class gpuSignal<double>;
