#pragma once
#include <isce3/core/forward.h>
#include <isce3/io/forward.h>

#include <complex>

namespace isce3 { namespace geocode {
/**
 * param[in,out] data a matrix of data that needs to be base-banded in azimuth
 * param[in] starting_range starting range of the data block 
 * param[in] sensing_start starting azimuth time of the data block 
 * param[in] range_pixel_spacing spacing of the slant range 
 * param[in] prf pulse repetition frequency
 * param[in] doppler_lut 2D LUT of the image Doppler
 */
void baseband(isce3::core::Matrix<std::complex<float>>& data,
              const double starting_range, const double sensing_start,
              const double range_pixel_spacing, const double prf,
              const isce3::core::LUT2d<double>& doppler_lut);

}} // namespace isce3::geocode
