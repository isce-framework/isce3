#pragma once

#include "forward.h"
#include <isce3/io/Raster.h>

namespace isce3 { namespace signal {

/** Returns a block from a complex raster upsampled in the
 * X (range) direction
 *
 * @param[in]  input_raster    Input complex raster
 * @param[out] output_array    Output upsampled complex array
 * @param[in]  offset_x        Offset in the X direction
 * @param[in]  offset_y        Offset in the Y direction
 * @param[in]  input_size_x    Size of the block in the X direction
 * @param[in]  input_size_y    Size of the block in the Y direction
 * @param[in]  band            Raster band to read (default: 1)
 * @param[in]  upsample_factor Upsample factor in the X direction
 */
template<class T_val>
void upsampleRasterBlockX(isce3::io::Raster& input_raster,
                          std::valarray<std::complex<T_val>>& output_array,
                          size_t offset_x, size_t offset_y, size_t input_size_x,
                          size_t input_size_y, size_t band = 1,
                          size_t upsample_factor = 2);

}}