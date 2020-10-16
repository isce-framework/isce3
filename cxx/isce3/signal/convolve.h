#pragma once

#include "forward.h"

//#include <isce3/core/forward.h>
#include <isce3/io/forward.h>

#include <valarray>

#include <isce3/core/EMatrix.h>

namespace isce3 { namespace signal {

/** \brief 2D convolution in time domain with separable kernels. The full 2D
 * convolution is decomposed into two convolutions with 1D kernels whose product
 * is equivalent to convolution with 2D kernel. 
 * \param[out] output output data after convolution (row major) with shape of (nrows, ncols) where nrows equals size/ncols 
 * \param[in] input input data (row major) with shape of (nrows_padded, ncols_padded) where nrows_padded equals size/ncols_padded 
 * \param[in] kernelColumns 1D kernel in columns direction
 * \param[in] kernelRows 1D kernel in rows direction
 * \param[in] ncols number of columns in the input data before padding
 * \param[in] ncols_padded number of columns of the input data after padding (ncols_padded = ncols + 2*floor(kernelColumns.size()/2))
 * */
template<typename T>
void convolve2D(std::valarray<T>& output, const std::valarray<T>& input,
                const std::valarray<double>& kernelColumns,
                const std::valarray<double>& kernelRows, int ncols,
                int ncols_padded);

/** \brief 2D convolution in time domain with separable kernels. The full 2D
 * convolution is decomposed into two convolutions with 1D kernels whose product
 * is equivalent to convolution with 2D kernel. 
 * \param[out] output output data after convolution (row major) with shape of (nrows, ncols) where nrows equals size/ncols 
 * \param[in] input input data (row major) with shape of (nrows_padded, ncols_padded) where nrows_padded equals size/ncols_padded
 * \param[in] noDataValue the value which is considered as invalid and will be ignored during convolution 
 * \param[in] kernelColumns 1D kernel in columns direction 
 * \param[in] kernelRows 1D kernel in rows direction 
 * \param[in] ncols number of columns in the input data before padding 
 * \param[in] ncols_padded number of columns of the input data after padding (ncols_padded = ncols + 2*floor(kernelColumns.size()/2))
 * */
template<typename T>
void convolve2D(std::valarray<T>& output, const std::valarray<T>& input,
                const T& noDataValue,
                const std::valarray<double>& kernelColumns,
                const std::valarray<double>& kernelRows, int ncols,
                int ncols_padded);

/** \brief 2D convolution in time domain with separable kernels. The full 2D
convolution is decomposed into two convolutions with 1D kernels whose product is
equivalent to convolution with 2D kernel.

* \param[out] output output data after convolution (row major) with shape of (nrows, ncols) where nrows equals size/ncols
 * \param[in] input input data (row major) with shape of (nrows_padded, ncols_padded) where nrows_padded equals size/ncols_padded
 * \param[in] mask a binary mask to maks the input data. Pixels with false value are masked out.
 * \param[in] kernelColumns 1D kernel in columns direction
 * \param[in] kernelRows 1D kernel in rows direction
 * \param[in] ncols number of columns in the input data before padding
 * \param[in] ncols_padded number of columns of the input data after padding (ncols_padded = ncols + 2*floor(kernelColumns.size()/2))
 * */
template<typename T>
void convolve2D(std::valarray<T>& output, const std::valarray<T>& input,
                const std::valarray<bool>& mask,
                const std::valarray<double>& kernelColumns,
                const std::valarray<double>& kernelRows, int ncols,
                int ncols_padded);

/** \brief 2D convolution in time domain with separable kernels. The full 2D
 * convolution is decomposed into two convolutions with 1D kernels whose product
 * is equivalent to convolution with 2D kernel. 
 * \param[out] output output data after convolution (row major) with shape of (nrows, ncols) where nrows equals size/ncols 
 * \param[in] input input data (row major) with shape of (nrows_padded, ncols_padded) where nrows_padded equals size/ncols_padded
 * \param[in] weights to weight data before convolution with the shape of padded input data 
 * \param[in] kernelColumns 1D kernel in columns direction 
 * \param[in] kernelRows 1D kernel in rows direction 
 * \param[in] ncols number of columns in the input data before padding 
 * \param[in] ncols_padded number of columns of the input data after padding (ncols_padded = ncols + 2*floor(kernelColumns.size()/2))
 * */

template<typename T>
void convolve2D(std::valarray<T>& output, const std::valarray<T>& input,
                const std::valarray<double>& weights,
                const std::valarray<double>& kernelColumns,
                const std::valarray<double>& kernelRows, int ncols,
                int ncols_padded);

/** \brief 2D convolution in time domain with separable kernels. The full 2D
 * convolution is decomposed into two convolutions with 1D kernels whose product
 * is equivalent to convolution with 2D kernel. 
 * \param[out] output output data after convolution with the shape of input before padding) 
 * \param[in] input padded input data (shape of padded data = shape of data + 2*floor(shape of 2D kernel / 2)) 
 * \param[in] weights to weight data before convolution with the shape of padded input data 
 * \param[in] kernelColumns 1D kernel to be used for 1D convolution in columns direction 
 * \param[in] kernelRows 1D kernel to be used for 1D convolution in rows direction
 * */
template<typename T>
void convolve2D(isce3::core::EArray2D<T>& output,
                const isce3::core::EArray2D<T>& input,
                const isce3::core::EArray2D<double>& weights,
                const isce3::core::EArray2D<double>& kernelColumns,
                const isce3::core::EArray2D<double>& kernelRows);

}} // namespace isce3::signal
