#pragma once
#include <isce3/io/forward.h>

#include <valarray>

namespace isce3 { namespace signal {

/**
 * filters real or complex type data by convolving two 1D separable kernels in
 * columns and rows directions. 
 * \param[out] output_raster output filtered raster data 
 * \param[in] input_raster input raster data to be filtered 
 * \param[in] kernel_columns 1-D kernel in columns direction 
 * \param[in] kernel_rows 1-D kernel in rows direction 
 * \param[in] block_rows number of lines (rows) per block.
 */
template<typename T>
void filter2D(isce3::io::Raster& output_raster, isce3::io::Raster& input_raster,
              const std::valarray<double>& kernel_columns,
              const std::valarray<double>& kernel_rows, int block_rows = 1000);

/**
 * filters real or complex data by convolving two 1D separable kernels in
 * columns and rows directions. 
 * \param[out] output_raster output filtered raster data 
 * \param[in] input_raster input raster data to be filtered 
 * \param[in] mask_raster input mask to mask the data before filtering 
 * \param[in] kernel_columns 1-D kernel in columns direction 
 * \param[in] kernel_rows 1-D kernel in rows direction 
 * \param[in] block_rows number of lines (rows) per block
 * */
template<typename T>
void filter2D(isce3::io::Raster& output_raster, isce3::io::Raster& input_raster,
              isce3::io::Raster& mask_raster,
              const std::valarray<double>& kernel_columns,
              const std::valarray<double>& kernel_rows, int block_rows = 1000);

/**
 * filters real or complex data by convolving two 1D separable kernels in
 * columns and rows directions. 
 * \param[out] output_raster output filtered raster data 
 * \param[in] input_raster input raster data to be filtered 
 * \param[in] mask_raster input mask to mask the data before filtering 
 * \param[in] kernel_columns 1-D kernel in columns direction 
 * \param[in] kernel_rows 1-D kernel in rows direction 
 * \param[in] do_decimate flag to indicate if the output data will be decimated proportional to the kernel size 
 * \param[in] mask flag to indicate if the output data will be masked before filtering
 * \param[in] block_rows number of lines (rows) per block
 * */
template<typename T>
void filter2D(isce3::io::Raster& output_raster, isce3::io::Raster& input_raster,
              isce3::io::Raster& mask_raster,
              const std::valarray<double>& kernel_columns,
              const std::valarray<double>& kernel_rows, const bool do_decimate,
              const bool mask = true, int block_rows = 1000);

}} // namespace isce3::signal
