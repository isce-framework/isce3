#pragma once

#include <valarray>

namespace isce3 { namespace signal {

/**
 * Decimate a 2D dataset in x and y directions with different decimation factor.
 * \param[out] out output decimated data
 * \param[in] in input data to be decimated
 * \param[in] nrows number of rows in the input array before decimation
 * \param[in] ncols number of columns in the output array before decimation
 * \param[in] nrows_decimated number of rows in the decimated data  = (nrows - rows_offset - 1) / rows_decimation + 1 
 * \param[in] ncols_decimated number of columns in the decimated data  = (ncols - cols_offset - 1) / cols_decimation + 1 
 * \param[in] rows_decimation decimation factor in rows direction 
 * \param[in] cols_decimation decimation factor in columns direction 
 * \param[in] rows_offset offset in row direction to start decimation 
 * \param[in] cols_offset offset in columns direction to start decimation
 */
template<typename T>
void decimate(std::valarray<T>& out, const std::valarray<T>& in, size_t nrows,
              size_t ncols, size_t nrows_decimated, size_t ncols_decimated,
              size_t rows_decimation, size_t cols_decimation,
              size_t rows_offset = 0, size_t cols_offset = 0);

}} // namespace isce3::signal
