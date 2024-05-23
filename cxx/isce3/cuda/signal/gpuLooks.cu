#include "gpuLooks.h"
#include <isce3/cuda/except/Error.h>

#define THRD_PER_BLOCK 1024 // Number of threads per block (should always %32==0)

namespace isce3::cuda::signal {

/**
input:
hi_res
output:
lo_res
 */
template <typename T>
void gpuLooks<T>::multilook(std::valarray<T> &hi_res,
        std::valarray<T> &lo_res)
{
    // allocate lo res output on device
    T *d_lo_res;
    size_t n_lo_res_size = _nrowsLooked*_ncolsLooked;
    size_t lo_res_size = n_lo_res_size*sizeof(T);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lo_res), lo_res_size));

    // allocate and copy to device hi res input
    T *d_hi_res;
    size_t hi_res_size = _nrows*_ncols*sizeof(T);
    // allocate input
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_hi_res), hi_res_size));
    // copy hi_res
    checkCudaErrors(cudaMemcpy(d_hi_res, &hi_res[0], hi_res_size, cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_lo_res_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // run kernels
    multilooks_g<<<grid, block>>>(d_lo_res,
            d_hi_res,
            _ncols,
            _ncolsLooked,
            _rowsLooks,
            _colsLooks,
            _nrowsLooked*_ncolsLooked,
            T(_rowsLooks*_colsLooks));

    // copy from device lo res output
    checkCudaErrors(cudaMemcpy(&lo_res[0], d_lo_res, lo_res_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_lo_res));
    checkCudaErrors(cudaFree(d_hi_res));
}

/**
input:
hi_res
output:
lo_res
 */
template <typename T>
void gpuLooks<T>::multilook(std::valarray<std::complex<T>> &hi_res,
        std::valarray<std::complex<T>> &lo_res)
{
    // allocate lo res output on device
    thrust::complex<T> *d_lo_res;
    size_t n_lo_res_size = _nrowsLooked*_ncolsLooked;
    size_t lo_res_size = n_lo_res_size*sizeof(thrust::complex<T>);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lo_res), lo_res_size));

    // allocate and copy to device hi res input
    thrust::complex<T> *d_hi_res;
    size_t hi_res_size = _nrows*_ncols*sizeof(thrust::complex<T>);
    // allocate input
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_hi_res), hi_res_size));
    // copy hi_res
    checkCudaErrors(cudaMemcpy(d_hi_res, &hi_res[0], hi_res_size, cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_lo_res_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // run kernels
    multilooks_g<<<grid, block>>>(d_lo_res,
            d_hi_res,
            _ncols,
            _ncolsLooked,
            _rowsLooks,
            _colsLooks,
            _nrowsLooked*_ncolsLooked,
            T(_rowsLooks*_colsLooks));

    // copy from device lo res output
    checkCudaErrors(cudaMemcpy(&lo_res[0], d_lo_res, lo_res_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_lo_res));
    checkCudaErrors(cudaFree(d_hi_res));
}

template <typename T>
void gpuLooks<T>::multilook(std::valarray<T> &hi_res,
        std::valarray<T> &lo_res,
        T noDataValue)
{
    // allocate lo res output on device
    T *d_lo_res;
    size_t n_lo_res_size = _nrowsLooked*_ncolsLooked;
    size_t lo_res_size = n_lo_res_size*sizeof(T);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lo_res), lo_res_size));

    // allocate and copy to device hi res input
    T *d_hi_res;
    size_t hi_res_size = _nrows*_ncols*sizeof(T);
    // allocate input
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_hi_res), hi_res_size));
    // copy hi_res
    checkCudaErrors(cudaMemcpy(d_hi_res, &hi_res[0], hi_res_size, cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_lo_res_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // run kernels
    multilooks_no_data_g<<<grid, block>>>(d_lo_res,
            d_hi_res,
            noDataValue,
            _ncols,
            _ncolsLooked,
            _rowsLooks,
            _colsLooks,
            _nrowsLooked*_ncolsLooked,
            T(_rowsLooks*_colsLooks));

    // copy from device lo res output
    checkCudaErrors(cudaMemcpy(&lo_res[0], d_lo_res, lo_res_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_lo_res));
    checkCudaErrors(cudaFree(d_hi_res));
}

template <typename T>
void gpuLooks<T>::multilook(std::valarray<std::complex<T>> &hi_res,
        std::valarray<std::complex<T>> &lo_res,
        std::complex<T> noDataValue)
{
    // allocate lo res output on device
    thrust::complex<T> *d_lo_res;
    size_t n_lo_res_size = _nrowsLooked*_ncolsLooked;
    size_t lo_res_size = n_lo_res_size*sizeof(thrust::complex<T>);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lo_res), lo_res_size));

    // allocate and copy to device hi res input
    thrust::complex<T> *d_hi_res;
    size_t hi_res_size = _nrows*_ncols*sizeof(thrust::complex<T>);
    // allocate input
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_hi_res), hi_res_size));
    // copy hi_res
    checkCudaErrors(cudaMemcpy(d_hi_res, &hi_res[0], hi_res_size, cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_lo_res_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // run kernels
    multilooks_no_data_g<<<grid, block>>>(d_lo_res,
            d_hi_res,
            thrust::complex<T>(noDataValue),
            _ncols,
            _ncolsLooked,
            _rowsLooks,
            _colsLooks,
            _nrowsLooked*_ncolsLooked,
            T(_rowsLooks*_colsLooks));

    // copy from device lo res output
    checkCudaErrors(cudaMemcpy(&lo_res[0], d_lo_res, lo_res_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_lo_res));
    checkCudaErrors(cudaFree(d_hi_res));
}

template <typename T>
void gpuLooks<T>::multilook(std::valarray<T> &hi_res,
        std::valarray<T> &lo_res,
        std::valarray<T> &weights)
{
    // allocate lo res output on device
    T *d_lo_res;
    size_t n_lo_res_size = _nrowsLooked*_ncolsLooked;
    size_t lo_res_size = n_lo_res_size*sizeof(T);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lo_res), lo_res_size));

    // allocate and copy to device hi res input
    T *d_hi_res;
    size_t hi_res_size = _nrows*_ncols*sizeof(T);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_hi_res), hi_res_size));
    checkCudaErrors(cudaMemcpy(d_hi_res, &hi_res[0], hi_res_size, cudaMemcpyHostToDevice));

    // allocate and copy to device weights
    T *d_weights;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_weights), hi_res_size));
    checkCudaErrors(cudaMemcpy(d_weights, &weights[0], hi_res_size, cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_lo_res_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // run kernels
    multilooks_weighted_g<<<grid, block>>>(d_lo_res,
         d_hi_res,
         d_weights,
         _ncols,
         _ncolsLooked,
         _rowsLooks,
         _colsLooks,
         _nrowsLooked*_ncolsLooked);

    // copy from device lo res output
    checkCudaErrors(cudaMemcpy(&lo_res[0], d_lo_res, lo_res_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_lo_res));
    checkCudaErrors(cudaFree(d_hi_res));
    checkCudaErrors(cudaFree(d_weights));
}

template <typename T>
void gpuLooks<T>::multilook(std::valarray<std::complex<T>> &hi_res,
        std::valarray<T> &lo_res,
        int p)
{
    // allocate lo res output on device
    T *d_lo_res;
    size_t n_lo_res_size = _nrowsLooked*_ncolsLooked;
    size_t lo_res_size = n_lo_res_size*sizeof(T);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_lo_res), lo_res_size));

    // allocate and copy to device hi res input
    thrust::complex<T> *d_hi_res;
    size_t hi_res_size = _nrows*_ncols*sizeof(thrust::complex<T>);
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_hi_res), hi_res_size));
    checkCudaErrors(cudaMemcpy(d_hi_res, &hi_res[0], hi_res_size, cudaMemcpyHostToDevice));

    // determine block layout
    dim3 block(THRD_PER_BLOCK);
    dim3 grid((n_lo_res_size+(THRD_PER_BLOCK-1))/THRD_PER_BLOCK);

    // run kernels
    multilooks_power_g<<<grid, block>>>(d_lo_res,
            d_hi_res,
            p,
            _ncols,
            _ncolsLooked,
            _rowsLooks,
            _colsLooks,
            _nrowsLooked*_ncolsLooked,
            T(_rowsLooks*_colsLooks));

    // copy from device lo res output
    checkCudaErrors(cudaMemcpy(&lo_res[0], d_lo_res, lo_res_size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_lo_res));
    checkCudaErrors(cudaFree(d_hi_res));
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
__global__ void multilooks_g(T *lo_res,
        const T* __restrict__ hi_res,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < sz_lo) {
        auto i_lo_row = i / n_cols_lo;
        auto i_lo_col = i % n_cols_lo;

        // init mlook accumulation to 0
        T accumulation = 0.0;

        // loop over contributing hi_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            auto i_hi_row = i_blk_row + i_lo_row * row_resize;
            // loop over contributing hi_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                auto i_hi_col = i_blk_col + i_lo_col * col_resize;
                // combine lo_res row and col index to hi_res index
                auto i_hi = i_hi_row * n_cols_hi + i_hi_col;
                // accumulate lo_res into lo_res
                accumulation += hi_res[i_hi];
            }
        }
        lo_res[i] = accumulation / blk_sz;
    }
}

template <typename T>
__global__ void multilooks_g(thrust::complex<T> *lo_res,
        const thrust::complex<T>* __restrict__ hi_res,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < sz_lo) {
        auto i_lo_row = i / n_cols_lo;
        auto i_lo_col = i % n_cols_lo;

        // init mlook accumulation to 0
        thrust::complex<T> accumulation(0.0, 0.0);

        // loop over contributing hi_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            auto i_hi_row = i_blk_row + i_lo_row * row_resize;
            // loop over contributing hi_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                auto i_hi_col = i_blk_col + i_lo_col * col_resize;
                // combine lo_res row and col index to hi_res index
                auto i_hi = i_hi_row * n_cols_hi + i_hi_col;
                // accumulate lo_res into lo_res
                accumulation += hi_res[i_hi];
            }
        }
        lo_res[i] = accumulation / blk_sz;
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
__global__ void multilooks_no_data_g(T *lo_res,
        const T* __restrict__ hi_res,
        T no_data_value,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < sz_lo) {
        auto i_lo_row = i / n_cols_lo;
        auto i_lo_col = i % n_cols_lo;

        T accumulation = 0;
        int n_no_val = 0;

        // loop over contributing hi_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            auto i_hi_row = i_blk_row + i_lo_row * row_resize;
            // loop over contributing hi_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                auto i_hi_col = i_blk_col + i_lo_col * col_resize;
                // combine lo_res row and col index to hi_res index
                auto i_hi = i_hi_row * n_cols_hi + i_hi_col;
                // accumulate lo_res into lo_res
                T hi_res_pixel_value = hi_res[i_hi];
                if (hi_res_pixel_value != no_data_value)
                    accumulation += hi_res_pixel_value;
                else
                    ++n_no_val;
            }
        }
        lo_res[i] = accumulation / (blk_sz - n_no_val);
    }
}

template <class T>
__global__ void multilooks_no_data_g(thrust::complex<T> *lo_res,
        const thrust::complex<T>* __restrict__ hi_res,
        thrust::complex<T> no_data_value,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < sz_lo) {
        auto i_lo_row = i / n_cols_lo;
        auto i_lo_col = i % n_cols_lo;

        thrust::complex<T> accumulation(0.0, 0.0);
        int n_no_val = 0;

        // loop over contributing hi_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            auto i_hi_row = i_blk_row + i_lo_row * row_resize;
            // loop over contributing hi_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                auto i_hi_col = i_blk_col + i_lo_col * col_resize;
                // combine lo_res row and col index to hi_res index
                auto i_hi = i_hi_row * n_cols_hi + i_hi_col;
                // accumulate lo_res into lo_res
                thrust::complex<T> hi_res_pixel_value = hi_res[i_hi];
                if (hi_res_pixel_value != no_data_value)
                    accumulation += hi_res_pixel_value;
                else
                    ++n_no_val;
            }
        }
        lo_res[i] = accumulation / (blk_sz - n_no_val);
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
__global__ void multilooks_weighted_g(T *lo_res,
         const T* __restrict__ hi_res,
         const T* __restrict__ weights,
         size_t n_cols_hi,
         size_t n_cols_lo,
         int row_resize,
         int col_resize,
         size_t sz_lo)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < sz_lo) {
        auto i_lo_row = i / n_cols_lo;
        auto i_lo_col = i % n_cols_lo;

        T accumulation = 0;
        T sum_weight = 0;
        // loop over contributing hi_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            auto i_hi_row = i_blk_row + i_lo_row * row_resize;
            // loop over contributing hi_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                auto i_hi_col = i_blk_col + i_lo_col * col_resize;
                // combine lo_res row and col index to hi_res index
                auto i_hi = i_hi_row * n_cols_hi + i_hi_col;
                // accumulate lo_res into lo_res
                accumulation += hi_res[i_hi];
                sum_weight += weights[i_hi];
            }
        }

        if (sum_weight > 0) {
            lo_res[i] = accumulation / sum_weight;
        } else {
            lo_res[i] = 0.0;
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
__global__ void multilooks_power_g(T *lo_res,
        const thrust::complex<T>* __restrict__ hi_res,
        int power,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        T blk_sz)
{
    const auto i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < sz_lo) {
        auto i_lo_row = i / n_cols_lo;
        auto i_lo_col = i % n_cols_lo;

        // init mlook accumulation to 0
        T accumulation = 0.0;

        // loop over contributing hi_res rows
        for (int i_blk_row = 0; i_blk_row < row_resize; ++i_blk_row) {
            // get lo_res row index
            auto i_hi_row = i_blk_row + i_lo_row * row_resize;
            // loop over contributing hi_res columns
            for (int i_blk_col = 0; i_blk_col < col_resize; ++i_blk_col) {
                // get lo_res col index
                auto i_hi_col = i_blk_col + i_lo_col * col_resize;
                // combine lo_res row and col index to hi_res index
                auto i_hi = i_hi_row * n_cols_hi + i_hi_col;
                // accumulate lo_res into lo_res
                accumulation += pow(abs(hi_res[i_hi]), power);
            }
        }
        lo_res[i] = accumulation / blk_sz;
    }
}

/*
   declarations!
*/
template class gpuLooks<float>;

template __global__ void
multilooks_g<float>(float *lo_res,
        const float* __restrict__ hi_res,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        float blk_sz);

template __global__ void
multilooks_g<float>(thrust::complex<float> *lo_res,
        const thrust::complex<float>* __restrict__ hi_res,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        float blk_sz);

template __global__ void
multilooks_no_data_g<float>(float *lo_res,
        const float* __restrict__ hi_res,
        float no_data_value,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        float blk_sz);

template __global__ void
multilooks_no_data_g<float>(thrust::complex<float> *lo_res,
        const thrust::complex<float>* __restrict__ hi_res,
        thrust::complex<float> no_data_value,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        float blk_sz);

template __global__ void
multilooks_power_g<float>(float *lo_res,
        const thrust::complex<float>* __restrict__ hi_res,
        int power,
        size_t n_cols_hi,
        size_t n_cols_lo,
        int row_resize,
        int col_resize,
        size_t sz_lo,
        float blk_sz);

} // namespace isce3::cuda::signal
