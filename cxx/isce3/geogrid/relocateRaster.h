#pragma once

#include <isce3/core/Constants.h>
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>
#include <isce3/product/forward.h>

namespace isce3 { namespace geogrid {

/** Relocate raster
 *
 * Interpolate a raster file over a given geogrid. 
 * Invalid pixels are filled with NaN. The output raster is expected to
 * have the same length & width as the specified geogrid, and the same
 * number of bands as the input raster.
 *
 * @param[in]  input_raster                Input raster
 * @param[in]  geogrid                     Output DEM geogrid
 * @param[out] output_raster               Output raster
 * @param[in]  interp_method               Interpolation method
 */
void relocateRaster(
        isce3::io::Raster& input_raster,
        const isce3::product::GeoGridParameters& geogrid,
        isce3::io::Raster& output_raster,
        isce3::core::dataInterpMethod interp_method =
                    isce3::core::dataInterpMethod::BIQUINTIC_METHOD);

}}