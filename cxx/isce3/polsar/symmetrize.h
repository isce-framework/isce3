#pragma once

#include <isce3/core/Constants.h>
#include <isce3/io/Raster.h>

namespace isce3 { namespace polsar {

/** Symmetrize cross-polarimetric channels.
 *
 * The current implementation considers that the cross-polarimetric
 * channels (HV and VH) are already calibrated so the polarimetric
 * symmetrization is represented by the simple arithmetic average.
 *
 * @param[in]  hv_raster           Raster containing the HV polarization
 * channel
 * @param[in]  vh_raster           Raster containing the VH polarization
 * channel
 * @param[out] output_raster       Output symmetrized raster
 * @param[in]  memory_mode         Memory mode. Option AutoBlocksY (default)
 * is equivalent to MultipleBlocksY (i.e. processing with multiple blocks
 * in the Y-direction).
 * @param[in]  hv_raster_band      Band (starting from 1) containing the HV
 * polarization channel within the `hv_raster`
 * @param[in]  vh_raster_band      Band (starting from 1) containing the VH
 * polarization channel within the `vh_raster`
 * @param[in]  output_band         Band (starting from 1) that will contain
 * the symmetrized cross-polarimetric channel
 */
void symmetrizeCrossPolChannels(isce3::io::Raster& hv_raster_band,
        isce3::io::Raster& vh_raster, isce3::io::Raster& output_raster,
        isce3::core::MemoryModeBlockY memory_mode =
                isce3::core::MemoryModeBlockY::AutoBlocksY,
        int hv_band = 1, int vh_raster_band = 1, int output_band = 1);

}} // namespace isce3::polsar