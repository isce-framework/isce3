#include <complex>
#include <Eigen/Core>
#include <limits>

#include <isce3/core/LUT2d.h>
#include <isce3/image/Resample.h>
#include <isce3/product/RadarGridParameters.h>

namespace isce3::cuda::image::v2 {

using isce3::image::v2::ArrayRef2D;
using isce3::image::v2::ConstArrayRef2D;

/** Interpolate input SLC block into the index values of the output block.
 *
 * @param[out] resampled_data_block
 * block of data in alternative radar coordinates
 * @param[in] input_data_block
 * block of SLC data in radar coordinates basebanded in range direction
 * @param[in] range_input_indices
 * range (radar-coordinates x) index of the pixels in input grid
 * @param[in] azimuth_input_indices
 * azimuth (radar-coordinates y) index of the pixels in input grid
 * @param[in] radar_grid
 * RadarGridParameters of input radar data
 * @param[in] native_doppler_lut
 * native doppler of input SLC image, in hz
 * @param[in] fill_value
 * value assigned to out-of-bounds pixels; defaults to NaN
 */
void gpuResampleToCoords(
    ArrayRef2D<std::complex<float>> resampled_data_block,
    const ConstArrayRef2D<std::complex<float>> input_data_block,
    const ConstArrayRef2D<double> range_input_indices,
    const ConstArrayRef2D<double> azimuth_input_indices,
    const isce3::product::RadarGridParameters& radar_grid,
    const isce3::core::LUT2d<double>& native_doppler_lut,
    const std::complex<float> fill_value = std::complex<float>(
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()
    )
);

} // namespace isce3::cuda::image::v2
