#pragma once
#include <isce3/io/forward.h>

#include <complex>
#include <iostream>

#include <isce3/core/Interpolator.h>
#include <isce3/core/Matrix.h>

namespace isce3 { namespace geocode {

/**
 * @param[in] rdrDataBlock a basebanded block of data in radar coordinate
 * @param[out] geoDataBlock a block of data in geo coordinates
 * @param[in] radarX the radar-coordinates x-index of the pixels in geo-grid
 * @param[in] radarY the radar-coordinates y-index of the pixels in geo-grid
 * @param[in] geometricalPhase the geometrical phase of each pixel in geo-grid
 * @param[in] radarBlockWidth width of the data block in radar coordinates
 * @param[in] radarBlockLength length of the data block in radar coordinates
 * @param[in] azimuthFirstLine azimuth time of the first sample
 * @param[in] rangeFirstPixel  range of the first sample
 * @param[in] interp interpolator object
 */
void interpolate(const isce3::core::Matrix<std::complex<float>>& rdrDataBlock,
                 isce3::core::Matrix<std::complex<float>>& geoDataBlock,
                 const std::valarray<double>& radarX,
                 const std::valarray<double>& radarY,
                 const std::valarray<std::complex<double>>& geometricalPhase,
                 const int radarBlockWidth, const int radarBlockLength,
                 const int azimuthFirstLine, const int rangeFirstPixel,
                 const isce3::core::Interpolator<std::complex<float>>* interp);

}} // namespace isce3::geocode
