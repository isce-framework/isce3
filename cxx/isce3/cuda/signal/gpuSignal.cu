#include <cstdio>
#include <string>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include "gpuSignal.h"
#include <isce3/cuda/except/Error.h>

#define THRD_PER_BLOCK 1024 // Number of threads per block (should always %32==0)

namespace isce3::cuda::signal {

/** copy left half columns of lo-res to left most columns of hi-res
*  copy right half columns of lo-res to right most columns of hi-res
*  0 fill columns in hi-res that has nothing copied to it
*  e.g. lo-res 2x4 -> hi-res 2x6
*       1 1 1 1       1 1 0 0 1 1
*       1 1 1 1       1 1 0 0 1 1
*  @param[out] pointer to hi res data
*  @param[in] pointer to lo res data
*  @param[in] number of rows
*  @param[in] number of lo res columns
*  @param[in] number of hi res columns
*/
template<class T>
__global__ void rangeShift_g(thrust::complex<T>*data_hi_res,
                             const thrust::complex<T>* __restrict__ data_lo_res,
                             int n_rows,
                             int n_cols_lo,
                             int n_cols_hi)
{
    // determine 1-d index of hi-res array
    const auto i_hi = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i_hi < n_cols_hi * n_rows) {
        // determine row and column from 1-d index
        auto i_row = i_hi / n_cols_hi;
        auto i_col = i_hi % n_cols_hi;

        // check if lo column is on the left
        if (i_col < (n_cols_lo + 1)/ 2) {
            auto i_lo = i_row * n_cols_lo + i_col;
            data_hi_res[i_hi] = data_lo_res[i_lo];
        // check if lo column is on the right
        } else if (i_col > n_cols_hi - n_cols_lo / 2) {
            auto i_lo = i_row * n_cols_lo + i_col - (n_cols_hi - n_cols_lo);
            data_hi_res[i_hi] = data_lo_res[i_lo];
        // else zero in the middle
        } else {
            data_hi_res[i_hi] = thrust::complex<T>(0.0, 0.0);
        }
    }
}

/** apply complex shift impact
*  @param[in] pointer to data to be multiplied by shift impact
*  @param[in] pointer to shift impact data
*  @param[in] number of total elements to be multiplied
*/
template<class T>
__global__ void shiftImpact_g(thrust::complex<T> *data,
                              const thrust::complex<T>* __restrict__ shift_impact,
                              size_t n_elements)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n_elements) {
        data[i] *= shift_impact[i];
    }
}

/** normalize in-place on device
*  @param[in] pointer to data to be normalized
*  @param[in] normalization factor
*  @param[in] number of total elements to be normalized
*/
template<class T>
__global__ void normalize_g(thrust::complex<T> *data,
                            const T normalization,
                            size_t n_elements)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n_elements) {
        data[i] /= normalization;
    }
}


/** Constructor **/
template<class T>
gpuSignal<T>::
gpuSignal(cufftType _type) {
    _cufft_type = _type;
    _plan_set = false;
    _d_data = NULL;
    _d_data_set = false;
}


/** Destructor **/
template<class T>
gpuSignal<T>::
~gpuSignal() {
    if (_plan_set) {
        cufftDestroy(_plan);
    }

    if (_d_data_set) {
        checkCudaErrors(cudaFree(_d_data));
    }
}


/** sets up range 1D FFT
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


/** sets up azimuth 1D FFT
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


/** sets up 2D FFT
*  @param[in] ncolumns number of columns of the block of data
*  @param[in] nrows number of rows of the block of data
 */
template<class T>
void gpuSignal<T>::
FFT2D(int ncolumns, int nrows)
{
    _n_elements = nrows * ncolumns;
    if (_plan_set) {
        cufftDestroy(_plan);
    }

    checkCudaErrors(cufftCreate(&_plan));
    _plan_set = true;
    size_t worksize;
    checkCudaErrors(cufftMakePlan2d(_plan, nrows, ncolumns, _cufft_type, &worksize));
}


/** sets up 1D cufft
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
    if (_plan_set) {
        cufftDestroy(_plan);
    }

    checkCudaErrors(cufftCreate(&_plan));
    _plan_set = true;
    size_t worksize;
    checkCudaErrors(cufftMakePlanMany(_plan, rank, n,
                                      inembed, istride, idist,
                                      onembed, ostride, odist,
                                      _cufft_type, _howmany, &worksize));
}


/** finds next power of 2 >= given size
*   @param[in] N the actual length of a signal
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


/** sets range specific FFT parameters
*   @param[in] ncolumns number of columns
*   @param[in] nrows number of rows
*/
template <class T>
void gpuSignal<T>::
_configureRangeFFT(int ncolumns, int nrows)
{
    _rank = 1;
    _n[0] = ncolumns;
    _n[1] = 0;

    _howmany = nrows;

    _inembed[0] = ncolumns;
    _inembed[0] = 0;

    _istride = 1;
    _idist = ncolumns;

    _onembed[0] = ncolumns;
    _onembed[1] = 0;

    _ostride = 1;
    _odist = ncolumns;

    _n_elements = nrows * ncolumns;

    _rows = nrows;
    _columns = ncolumns;
}


/** sets up azimuth specific FFT parameters
*   @param[in] ncolumns number of columns
*   @param[in] nrows number of rows
*/
template <class T>
void gpuSignal<T>::
_configureAzimuthFFT(int ncolumns, int nrows)
{
    _rank = 1;
    _n[0] = nrows;
    _n[1] = 0;

    _howmany = ncolumns;

    _inembed[0] = nrows;
    _inembed[1] = 0;

    _istride = ncolumns;
    _idist = 1;

    _onembed[0] = nrows;
    _onembed[1] = 0;

    _ostride = ncolumns;
    _odist = 1;

    _n_elements = nrows * ncolumns;

    _rows = nrows;
    _columns = ncolumns;
}


/** copies data from host to device
*   @param[in] pointer to host data
*/
template<class T>
void gpuSignal<T>::
dataToDevice(std::complex<T> *input)
{
    if (!_d_data_set) {
        size_t input_size = _n_elements*sizeof(thrust::complex<T>);
        // allocate input
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&_d_data), input_size));
        // copy input
        checkCudaErrors(cudaMemcpy(_d_data, input, input_size, cudaMemcpyHostToDevice));
        _d_data_set = true;
    }
}


/** copies data from host to device
*   @param[in] valarray on host data
*/
template<class T>
void gpuSignal<T>::
dataToDevice(std::valarray<std::complex<T>> &input)
{
    if (!_d_data_set) {
        size_t input_size = input.size()*sizeof(thrust::complex<T>);
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
        size_t output_size = _n_elements*sizeof(thrust::complex<T>);
        // copy output
        checkCudaErrors(cudaMemcpy(output, _d_data, output_size, cudaMemcpyDeviceToHost));
    }
}

template<class T>
void gpuSignal<T>::
dataToHost(std::valarray<std::complex<T>> &output)
{
    if (_d_data_set) {
        size_t output_size = _n_elements*sizeof(thrust::complex<T>);
        // copy output
        checkCudaErrors(cudaMemcpy(&output[0], _d_data, output_size, cudaMemcpyDeviceToHost));
    }
}


/** unnormalized forward complex float transform performed in place on class data
*/
template<class T>
void gpuSignal<T>::
forward()
{
    if (_plan_set && _d_data_set) {
        forwardDevMem(_d_data);
    }
}


/** unnormalized forward complex float transform performed on given device data
*   @param[in] pointer to source data on device
*   @param[out] pointer to output data on device
*/
template<>
void gpuSignal<float>::
forwardDevMem(thrust::complex<float> *input, thrust::complex<float> *output)
{
    // transform
    if (_plan_set) {
        cufftComplex* _input = reinterpret_cast<cufftComplex*>(input);
        cufftComplex* _output = reinterpret_cast<cufftComplex*>(output);
        checkCudaErrors(cufftExecC2C(_plan, _input, _output, CUFFT_FORWARD));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}


/** unnormalized forward complex double transform performed on given device data
*   @param[in] pointer to source data on device
*   @param[out] pointer to output data on device
*/
template<>
void gpuSignal<double>::
forwardDevMem(thrust::complex<double> *input, thrust::complex<double> *output)
{
    // transform
    if (_plan_set) {
        cufftDoubleComplex* _input = reinterpret_cast<cufftDoubleComplex*>(input);
        cufftDoubleComplex* _output = reinterpret_cast<cufftDoubleComplex*>(output);
        checkCudaErrors(cufftExecZ2Z(_plan, _input, _output, CUFFT_FORWARD));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}


/** unnormalized forward complex transform performed in place on given device data
*   @param[in] pointer to source/output data on device
*/
template<class T>
void gpuSignal<T>::
forwardDevMem(thrust::complex<T> *dataInPlace) {
    forwardDevMem(dataInPlace, dataInPlace);
}


/** unnormalized forward complex float transform
*  @param[in] pointer to input block of data
*  @param[in] pointer to output block of spectrum
*/
template<class T>
void gpuSignal<T>::
forwardC2C(std::complex<T> *input, std::complex<T> *output)
{
    if (_plan_set) {
        size_t input_size = _n_elements*sizeof(std::complex<T>);
        size_t output_size = _n_elements*sizeof(std::complex<T>);

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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(output, d_input, input_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** unnormalized forward complex float transform
*  @param[in] valarray containing input block of data
*  @param[in] valarray containing output block of spectrum
*/
template<class T>
void gpuSignal<T>::
forwardC2C(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    if (_plan_set) {
        size_t input_size = input.size()*sizeof(std::complex<T>);
        size_t output_size = output.size()*sizeof(std::complex<T>);

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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(&output[0], d_input, input_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** unnormalized forward complex double transform
*  @param[in] pointer to input block of data
*  @param[in] pointer to output block of spectrum
*/
template<class T>
void gpuSignal<T>::
forwardZ2Z(std::complex<T> *input, std::complex<T> *output)
{
    if (_plan_set) {
        size_t input_size = _n_elements*sizeof(std::complex<T>);
        size_t output_size = _n_elements*sizeof(std::complex<T>);

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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(output, d_input, input_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** unnormalized forward complex double transform
*  @param[in] valarray containing input block of data
*  @param[in] valarray containing output block of spectrum
*/
template<class T>
void gpuSignal<T>::
forwardZ2Z(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    if (_plan_set) {
        size_t input_size = input.size()*sizeof(std::complex<T>);
        size_t output_size = output.size()*sizeof(std::complex<T>);

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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(&output[0], d_input, input_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** unnormalized forward double to complex double transform
*  @param[in] pointer to input block of data
*  @param[in] pointer to output block of spectrum
*/
template<class T>
void gpuSignal<T>::
forwardD2Z(T *input, std::complex<T> *output)
{
    if (_plan_set) {
        size_t input_size = _n_elements*sizeof(T);
        size_t output_size = _n_elements*sizeof(std::complex<T>);

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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** unnormalized forward complex float transform
*  @param[in] pointer to input block of data
*  @param[in] pointer to output block of spectrum
*/
template<>
void gpuSignal<float>::
forward(std::complex<float> *input, std::complex<float> *output)
{
    forwardC2C(input, output);
}


/** unnormalized forward complex double transform
*  @param[in] pointer to input block of data
*  @param[in] pointer to output block of spectrum
*/
template<>
void gpuSignal<double>::
forward(std::complex<double> *input, std::complex<double> *output)
{
    forwardZ2Z(input, output);
}


/** unnormalized forward complex float transform
*  @param[in] valarray containing input block of data
*  @param[in] valarray containing output block of spectrum
*/
template<>
void gpuSignal<float>::
forward(std::valarray<std::complex<float>> &input, std::valarray<std::complex<float>> &output)
{
    forwardC2C(input, output);
}


/** unnormalized forward complex double transform
*  @param[in] valarray containing input block of data
*  @param[in] valarray containing output block of spectrum
*/
template<>
void gpuSignal<double>::
forward(std::valarray<std::complex<double>> &input, std::valarray<std::complex<double>> &output)
{
    forwardZ2Z(input, output);
}


/** unnormalized inverse complex float transform performed in place on class data
*/
template<class T>
void gpuSignal<T>::
inverse()
{
    if (_plan_set && _d_data_set) {
        inverseDevMem(_d_data);
    }
}


/** unnormalized inverse complex float transform performed on given device data
*   @param[in] pointer to source data on device
*   @param[out] pointer to output data on device
*/
template<>
void gpuSignal<float>::
inverseDevMem(thrust::complex<float> *input, thrust::complex<float> *output)
{
    // transform
    if (_plan_set) {
        cufftComplex* _input = reinterpret_cast<cufftComplex*>(input);
        cufftComplex* _output = reinterpret_cast<cufftComplex*>(output);
        checkCudaErrors(cufftExecC2C(_plan, _input, _output, CUFFT_INVERSE));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}


/** unnormalized inverse complex double transform performed on given device data
*   @param[in] pointer to source data on device
*   @param[out] pointer to output data on device
*/
template<>
void gpuSignal<double>::
inverseDevMem(thrust::complex<double> *input, thrust::complex<double> *output)
{
    // transform
    if (_plan_set) {
        cufftDoubleComplex* _input = reinterpret_cast<cufftDoubleComplex*>(input);
        cufftDoubleComplex* _output = reinterpret_cast<cufftDoubleComplex*>(output);
        checkCudaErrors(cufftExecZ2Z(_plan, _input, _output, CUFFT_INVERSE));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}


/** unnormalized inverse complex transform performed in place on given device data
*   @param[in] pointer to source/output data on device
*/
template<class T>
void gpuSignal<T>::
inverseDevMem(thrust::complex<T> *dataInPlace) {
    inverseDevMem(dataInPlace, dataInPlace);
}


/** unnormalized inverse complex float transform
*  @param[in] pointer to input block of spectrum
*  @param[in] pointer to output block of data
*/
template<class T>
void gpuSignal<T>::
inverseC2C(std::complex<T> *input, std::complex<T> *output)
{
    if (_plan_set) {
        size_t input_size = _n_elements*sizeof(std::complex<T>);
        size_t output_size = _n_elements*sizeof(std::complex<T>);

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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(output, d_input, input_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }

}


/** unnormalized inverse complex float transform
*  @param[in] valarray containing input block of spectrum
*  @param[in] valarray containing output block of data
*/
template<class T>
void gpuSignal<T>::
inverseC2C(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    if (_plan_set) {
        size_t input_size = input.size()*sizeof(std::complex<T>);
        size_t output_size = output.size()*sizeof(std::complex<T>);

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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(&output[0], d_input, input_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** unnormalized inverse complex double transform
*  @param[in] pointer to input block of spectrum
*  @param[in] pointer to output block of data
*/
template<class T>
void gpuSignal<T>::
inverseZ2Z(std::complex<T> *input, std::complex<T> *output)
{
    if (_plan_set) {
        size_t input_size = _n_elements*sizeof(std::complex<T>);
        size_t output_size = _n_elements*sizeof(std::complex<T>);

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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(output, d_input, input_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** unnormalized inverse complex double transform
*  @param[in] valarray containing input block of spectrum
*  @param[in] valarray containing output block of data
*/
template<class T>
void gpuSignal<T>::
inverseZ2Z(std::valarray<std::complex<T>> &input, std::valarray<std::complex<T>> &output)
{
    if (_plan_set) {
        size_t input_size = input.size()*sizeof(std::complex<T>);
        size_t output_size = output.size()*sizeof(std::complex<T>);

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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(&output[0], d_input, input_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** unnormalized inverse complex double to double transform
*  @param[in] pointer to input block of spectrum
*  @param[in] pointer to output block of data
*/
template<class T>
void gpuSignal<T>::
inverseZ2D(std::complex<T> *input, T *output)
{
    if (_plan_set) {
        size_t input_size = _n_elements*sizeof(std::complex<T>);
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

        checkCudaErrors(cudaDeviceSynchronize());

        // copy output
        checkCudaErrors(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** unnormalized inverse complex float transform
*  @param[in] pointer to input block of spectrum
*  @param[in] pointer to output block of data
*/
template<>
void gpuSignal<float>::
inverse(std::complex<float> *input, std::complex<float> *output)
{
    inverseC2C(input, output);
}


/** unnormalized inverse complex double transform
*  @param[in] pointer to input block of spectrum
*  @param[in] pointer to output block of data
*/
template<>
void gpuSignal<double>::
inverse(std::complex<double> *input, std::complex<double> *output)
{
    inverseZ2Z(input, output);
}


/** unnormalized inverse complex float transform
*  @param[in] valarray containing input block of spectrum
*  @param[in] valarray containing output block of data
*/
template<>
void gpuSignal<float>::
inverse(std::valarray<std::complex<float>> &input, std::valarray<std::complex<float>> &output)
{
    inverseC2C(input, output);
}


/** unnormalized inverse complex double transform
*  @param[in] valarray containing input block of spectrum
*  @param[in] valarray containing output block of data
*/
template<>
void gpuSignal<double>::
inverse(std::valarray<std::complex<double>> &input, std::valarray<std::complex<double>> &output)
{
    inverseZ2Z(input, output);
}


/** normalized complex float column/range-wise upsampling
*  @param[in] valarray containing lo res data
*  @param[in] valarray containing hi res data
*  @param[in] rows in both lo and hi res data
*  @param[in] columns in lo res data
*  @param[in] upsample factor
*  @param[in] valarray containing shift impact
*/
template<class T>
void gpuSignal<T>::
upsample(std::valarray<std::complex<T>> &input,
         std::valarray<std::complex<T>> &output,
         int row,
         int col,
         int upsampleFactor,
         std::valarray<std::complex<T>> &shiftImpact)
{
    if (_plan_set) {
        size_t input_size = _n_elements*sizeof(thrust::complex<T>);
        size_t output_size = upsampleFactor * _n_elements*sizeof(thrust::complex<T>);

        // allocate device memory
        thrust::complex<T> *d_input;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
        thrust::complex<T> *d_output;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));

        // copy input
        checkCudaErrors(cudaMemcpy(d_input, &input[0], input_size, cudaMemcpyHostToDevice));

        // forward transform
        forwardDevMem(d_input);

        // determine block layout
        dim3 block(THRD_PER_BLOCK);
        dim3 grid((input_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

        // shift data prior to upsampling transform
        rangeShift_g<<<grid, block>>>(d_output, d_input,
                                      _rows, _columns,
                                      upsampleFactor*_columns);

        if (shiftImpact.size() == output.size()) {
            thrust::complex<T> *d_shift_impact;
            size_t shift_size = shiftImpact.size()*sizeof(thrust::complex<T>);
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_shift_impact), shift_size));
            checkCudaErrors(cudaMemcpy(d_shift_impact,
                                       &shiftImpact[0],
                                       shift_size,
                                       cudaMemcpyHostToDevice));

            shiftImpact_g<<<grid, block>>>(d_output, d_shift_impact, shift_size);

            checkCudaErrors(cudaFree(d_shift_impact));
        }

        // set inverse transform
        rangeFFT(upsampleFactor*col, 1);

        // inverse transformation
        inverseDevMem(d_output);

        // normalize
        normalize_g<<<grid, block>>>(d_output, static_cast<T>(_columns), _n_elements);

        // copy output
        checkCudaErrors(cudaMemcpy(&output[0], d_output, output_size, cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
    }
}


/** normalized complex column/range-wise upsampling
*  @param[in] valarray containing lo res data
*  @param[in] valarray containing hi res data
*  @param[in] rows in both lo and hi res data
*  @param[in] columns in lo res data
*  @param[in] upsample factor
*/
template<class T>
void gpuSignal<T>::
upsample(std::valarray<std::complex<T>> &input,
         std::valarray<std::complex<T>> &output,
         int row,
         int nfft,
         int upsampleFactor)
{
    std::valarray<std::complex<T>> shiftImpact(0);

    upsample(input, output,
            row,
            nfft,
            upsampleFactor,
            shiftImpact);
}

/** upsample performed on device
*  @param[in] forward signal object
*  @param[in] inverse signal object
*  @param[in] pointer to data to be upsampled
*  @param[in] pointer to upsampled data
*/
template<class T>
void upsample(gpuSignal<T> &fwd,
        gpuSignal<T> &inv,
        thrust::complex<T> *input,
        thrust::complex<T> *output)
{
    fwd.forwardDevMem(input);

    // determine block layout
    auto input_size = fwd.getNumElements();
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((input_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // shift data prior to upsampling transform
    rangeShift_g<<<grid, block>>>(output,
                                  input,
                                  fwd.getRows(),
                                  fwd.getColumns(),
                                  inv.getColumns());

    inv.inverseDevMem(output);

    // columns**2 because fwd transform not normalized
    normalize_g<<<grid, block>>>(
            output,
            static_cast<T>(inv.getColumns()*inv.getColumns()),
            inv.getNumElements());
}


/** upsample performed on device
*  @param[in] forward signal object
*  @param[in] inverse signal object
*  @param[in] pointer to data to be upsampled
*  @param[in] pointer to upsampled data
*  @param[in] pointer to shift impact data
*/
template<class T>
void upsample(gpuSignal<T> &fwd,
        gpuSignal<T> &inv,
        thrust::complex<T> *input,
        thrust::complex<T> *output,
        thrust::complex<T> *shiftImpact)
{
    fwd.forwardDevMem(input);

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    auto nInvElements = inv.getNumElements();
    dim3 grid((nInvElements + (THRD_PER_BLOCK - 1)) / THRD_PER_BLOCK);

    // shift data prior to upsampling transform
    rangeShift_g<T><<<grid, block>>>(output,
                                     input,
                                     fwd.getRows(),
                                     fwd.getColumns(),
                                     inv.getColumns());

    shiftImpact_g<T><<<grid, block>>>(output,
                                      shiftImpact,
                                      nInvElements);

    inv.inverseDevMem(output);

    // columns**2 because fwd transform not normalized
    normalize_g<T><<<grid, block>>>(output,
                                    static_cast<T>(fwd.getColumns()),
                                    nInvElements);
}


/** upsample performed on device
*  @param[in] forward signal object
*  @param[in] inverse signal object
*  @param[in] valarray containing data to be upsampled
*  @param[in] valarray containing upsampled data
*/
template<class T>
void upsample(gpuSignal<T> &fwd,
        gpuSignal<T> &inv,
        std::valarray<std::complex<T>> &input,
        std::valarray<std::complex<T>> &output)
{
    std::valarray<std::complex<T>> empty_shift(0);

    upsample(fwd,
            inv,
            input,
            output,
            empty_shift);
}


/** upsample performed on device
*  @param[in] forward signal object
*  @param[in] inverse signal object
*  @param[in] valarray containing data to be upsampled
*  @param[in] valarray containing upsampled data
*  @param[in] valarray containing shift impact data
*/
template<class T>
void upsample(gpuSignal<T> &fwd,
        gpuSignal<T> &inv,
        std::valarray<std::complex<T>> &input,
        std::valarray<std::complex<T>> &output,
        std::valarray<std::complex<T>> &shiftImpact)
{
    auto input_size = input.size()*sizeof(thrust::complex<T>);
    thrust::complex<T> *d_input;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), input_size));
    checkCudaErrors(cudaMemcpy(d_input, &input[0], input_size, cudaMemcpyHostToDevice));

    auto output_size = output.size()*sizeof(thrust::complex<T>);
    thrust::complex<T> *d_output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), output_size));
    checkCudaErrors(cudaMemcpy(d_output, &output[0], output_size, cudaMemcpyHostToDevice));

    if (shiftImpact.size() > 0) {
        auto shiftImpact_size = shiftImpact.size()*sizeof(thrust::complex<T>);
        thrust::complex<T> *d_shiftImpact;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_shiftImpact), shiftImpact_size));
        checkCudaErrors(cudaMemcpy(d_shiftImpact, &shiftImpact[0], shiftImpact_size, cudaMemcpyHostToDevice));
        upsample(fwd,
                inv,
                d_input,
                d_output,
                d_shiftImpact);
        checkCudaErrors(cudaFree(d_shiftImpact));
    } else {
        upsample(fwd,
                inv,
                d_input,
                d_output);
    }

    checkCudaErrors(cudaMemcpy(&output[0], d_output, output_size, cudaMemcpyDeviceToHost));

    output /= static_cast<T>(inv.getColumns());

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}


/*
explicit instantiations
 */
template class gpuSignal<float>;
template class gpuSignal<double>;

template void
upsample<float>(gpuSignal<float> &fwd,
        gpuSignal<float> &inv,
        std::valarray<std::complex<float>> &input,
        std::valarray<std::complex<float>> &output);

template void
upsample<double>(gpuSignal<double> &fwd,
        gpuSignal<double> &inv,
        std::valarray<std::complex<double>> &input,
        std::valarray<std::complex<double>> &output);

template<class T>
void upsample(gpuSignal<T> &fwd,
        gpuSignal<T> &inv,
        std::valarray<std::complex<T>> &input,
        std::valarray<std::complex<T>> &output);

template void upsample<float>(gpuSignal<float> &fwd,
        gpuSignal<float> &inv,
        thrust::complex<float> *input,
        thrust::complex<float> *output,
        thrust::complex<float> *shiftImpact);

template void upsample<double>(gpuSignal<double> &fwd,
        gpuSignal<double> &inv,
        thrust::complex<double> *input,
        thrust::complex<double> *output,
        thrust::complex<double> *shiftImpact);

} // namespace isce3::cuda::signal
