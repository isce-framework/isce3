#include <complex>
#include <Eigen/Core>
#include <limits>

#include <isce3/core/LUT2d.h>
#include <isce3/core/Poly2d.h>
#include <isce3/product/RadarGridParameters.h>

namespace isce3::image::v2 {

template<class T, int Options = Eigen::RowMajor>
using Array2D = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Options>;

template<class T, int Options = Eigen::RowMajor>
using ArrayRef2D = Eigen::Ref<Array2D<T, Options>>;

template<class T, int Options = Eigen::RowMajor>
using ConstArrayRef2D = Eigen::Ref<const Array2D<T, Options>>;

/**
 * Remove range and azimuth phase carrier from a block of input radar SLC data
 *
 * @param[out] out
 * Block of data to be written to. All data in block will be overwritten.
 * unit: phase (complex) array2D
 * @tparam[in] carrier_phase
 * azimuth carrier phase of the SLC data, in radian, as a function of azimuth and range.
 * This phase will be removed from the resampled image.
 * @param[in] radar_grid
 * parameters for the given radar grid
 * @param[in] input_azimuth_first_line
 * line index of the first sample of the block of input data with respect to the origin
 * of the full SLC scene
 * unit: azimuth row indices (int)
 * @param[in] input_range_first_pixel
 * pixel index of the first sample of the block of input data with respect to the origin
 * of the full SLC scene
 * unit: range column indices (int)
 * @param[in] conjugate
 * if true, modulate the conjugate of the phase.
 */
template <typename AzRgFunc = isce3::core::Poly2d>
void getModulationPhase(
    ArrayRef2D<std::complex<float>> out,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const size_t input_azimuth_first_line,
    const size_t input_range_first_pixel,
    const bool conjugate
);


/**
 * Add back range and azimuth phase carrier and simultaneously flatten the block of
 * resampled SLC
 *
 * @param[out] out              Block of data to be written to. All data in block will
 *                              be overwritten.
 * @tparam[in] carrier_phase    carrier phase of the SLC data, in radian, as a
 *                              function of azimuth and range
 * @param[in] radar_grid        parameters for the given radar grid
 * @param[in] azimuth_indices   azimuth index of each output coordinate pixel in the
 *                              resampling coordinate system
 * @param[in] range_indices     range index of each output coordinate pixel in the
 *                              resampling coordinate system
 * @param[in] conjugate         if true, modulate the conjugate of the phase.
 */
template <typename AzRgFunc = isce3::core::Poly2d>
void getModulationPhaseAtCoords(
    ArrayRef2D<std::complex<float>> out,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const ConstArrayRef2D<double> azimuth_indices,
    const ConstArrayRef2D<double> range_indices,
    const bool conjugate
);


/**
 * Remove range and azimuth phase carrier from a block of input radar SLC data
 *
 * @param[out] slc_data_block
 * Block of data to be modulated.
 * unit: phase (complex) array2D
 * @tparam[in] carrier_phase
 * azimuth carrier phase of the SLC data, in radian, as a function of azimuth and range.
 * This phase will be removed from the resampled image.
 * @param[in] radar_grid
 * parameters for the given radar grid
 * @param[in] input_azimuth_first_line
 * line index of the first sample of the block of input data with respect to the origin
 * of the full SLC scene
 * unit: azimuth row indices (int)
 * @param[in] input_range_first_pixel
 * pixel index of the first sample of the block of input data with respect to the origin
 * of the full SLC scene
 * unit: range column indices (int)
 * @param[in] conjugate
 * if true, modulate the conjugate of the phase.
 */
template <typename AzRgFunc = isce3::core::Poly2d>
void modulate(
    ArrayRef2D<std::complex<float>> slc_data_block,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const size_t input_azimuth_first_line,
    const size_t input_range_first_pixel,
    const bool conjugate
);


/**
 * Add back range and azimuth phase carrier and simultaneously flatten the block of
 * resampled SLC
 *
 * @param[out] slc_data_block   Block of data to be modulated.
 * @tparam[in] carrier_phase    carrier phase of the SLC data, in radian, as a
 *                              function of azimuth and range
 * @param[in] radar_grid        parameters for the given radar grid
 * @param[in] azimuth_indices   azimuth index of each output coordinate pixel in the
 *                              resampling coordinate system
 * @param[in] range_indices     range index of each output coordinate pixel in the
 *                              resampling coordinate system
 * @param[in] conjugate         if true, modulate the conjugate of the phase.
 */
template <typename AzRgFunc = isce3::core::Poly2d>
void modulateAtCoords(
    ArrayRef2D<std::complex<float>> slc_data_block,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const ConstArrayRef2D<double> azimuth_indices,
    const ConstArrayRef2D<double> range_indices,
    const bool conjugate
);


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
 * parameters for the given radar grid
 * @param[in] native_doppler_lut
 * native doppler of SLC image
 * @param[in] fill_value
 * value assigned to out-of-bounds pixels or block in alternative radar coordinates;
 * defaults to NaN
 */
void resampleToCoords(
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

} // namespace isce3::image::v2
