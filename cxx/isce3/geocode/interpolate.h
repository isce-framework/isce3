#pragma once
#include <isce3/io/forward.h>

#include <complex>
#include <iostream>

#include <isce3/core/Interpolator.h>
#include <isce3/core/Matrix.h>
#include <isce3/product/Product.h>
#include <isce3/product/RadarGridParameters.h>

namespace isce3 { namespace geocode {

/**
 * @param[in] rdrDataBlock a basebanded block of data in radar coordinate
 * @param[out] geoDataBlock a block of data in geo coordinates
 * @param[in] radarX the radar-coordinates x-index of the pixels in geo-grid
 * @param[in] radarY the radar-coordinates y-index of the pixels in geo-grid
 * @param[in] azimuthFirstLine line index of the first sample of the block
 * @param[in] rangeFirstPixel  pixel index of the first sample of the block
 * @param[in] interp interpolator object
 * @param[in] radarGrid RadarGridParameters
 * @param[in] dopplerLUT Doppler LUT
 * @param[in] flatten flag to determine if the geocoded SLC will be flattened or
 * not
 */
void interpolate(const isce3::core::Matrix<std::complex<float>>& rdrDataBlock,
        isce3::core::Matrix<std::complex<float>>& geoDataBlock,
        const std::valarray<double>& radarX,
        const std::valarray<double>& radarY, const int azimuthFirstLine,
        const int rangeFirstPixel,
        const isce3::core::Interpolator<std::complex<float>>* interp,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::LUT2d<double>& dopplerLUT, const bool& flatten);

}} // namespace isce3::geocode
