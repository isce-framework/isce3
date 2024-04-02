#include <complex>
#include <Eigen/Core>
#include <limits>

#include <isce3/core/LUT2d.h>
#include <isce3/core/Poly2d.h>
#include <isce3/product/RadarGridParameters.h>

namespace isce3::image::modulate {

template<class T, int Options = Eigen::RowMajor>
using Array2D = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Options>;

template<class T, int Options = Eigen::RowMajor>
using ArrayRef2D = Eigen::Ref<Array2D<T, Options>>;

template<class T, int Options = Eigen::RowMajor>
using ConstArrayRef2D = Eigen::Ref<const Array2D<T, Options>>;

/**
 * Acquire the phase of the given carrier of a radar scene.
 *
 * @param[out] out
 * Block of data to be written to. All data in block will be overwritten.
 * unit: array2D of complex
 * @tparam[in] carrier_phase
 * azimuth carrier phase of the SLC data, ::modulate, as a function of azimuth and range.
 * @param[in] radar_grid
 * parameters for the given radar grid corresponding to the output block.
 * @param[in] conjugate
 * if true, get the conjugate of the phase.
 * @param[in] fill_value
 * The value to fill out-of-bounds pixels with. Out-of-bounds pixels are defined here as
 * any pixels that cannot be evaluated by the carrier_phase function. Poly2d functions
 * do not have out-of-bounds pixels, but LUT2d functions may. Defaults to NaN + j*NaN.
 */
template <typename AzRgFunc = isce3::core::Poly2d>
void getModulationPhase(
    ArrayRef2D<std::complex<float>> out,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const bool conjugate,
    const std::complex<float> fill_value
);


/**
 * Acquire the phase of the given carrier at each given index of a radar scene.
 * 
 * Note that this function does not perform size checks between `out` and the indices
 * blocks and radar grid.
 *
 * @param[out] out
 * Block of data to be written to. All data in block will be overwritten.
 * unit: array2D of complex
 * @tparam[in] carrier_phase
 * azimuth carrier phase of the SLC data, in radians, as a function of azimuth and range.
 * @param[in] radar_grid
 * parameters for the given radar grid corresponding to the output block.
 * @param[in] azimuth_indices
 * azimuth index of each pixel in the output block w.r.t the radar grid. Must be the
 * same shape as `out`.
 * unit: azimuth row indices (int)
 * @param[in] range_indices
 * range index of each pixel in the output block w.r.t the radar grid. Must be the
 * same shape as `out`.
 * unit: range column indices (int)
 * @param[in] conjugate
 * if true, modulate the conjugate of the phase.
 * @param[in] fill_value
 * The value to fill out-of-bounds pixels with. Out-of-bounds pixels are defined here as
 * any pixels that cannot be evaluated by the carrier_phase function. Poly2d functions
 * do not have out-of-bounds pixels, but LUT2d functions may. Defaults to NaN + j*NaN.
 */
template <typename AzRgFunc = isce3::core::Poly2d>
void _getModulationPhaseAtCoords(
    ArrayRef2D<std::complex<float>> out,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const ConstArrayRef2D<double> azimuth_indices,
    const ConstArrayRef2D<double> range_indices,
    const bool conjugate,
    const std::complex<float> fill_value
);


/**
 * Remove range and azimuth phase carrier from a block of input radar SLC data
 *
 * @param[out] slc_data_block
 * the block of data SLC to be modulated.
 * unit: array2D of complex
 * @tparam[in] carrier_phase
 * carrier phase of the SLC data, in radians, as a function of azimuth and range.
 * This phase will be modulated to or demodulated from the image.
 * @param[in] radar_grid
 * parameters for the given radar grid corresponding to `slc_data_block`.
 * @param[in] conjugate
 * if true, modulate the conjugate of the phase.
 * @param[in] fill_value
 * The value to fill out-of-bounds pixels with. Out-of-bounds pixels are defined here as
 * any pixels that cannot be evaluated by the carrier_phase function. Poly2d functions
 * do not have out-of-bounds pixels, but LUT2d functions may. Defaults to NaN + j*NaN.
 */
template <typename AzRgFunc = isce3::core::Poly2d>
void modulate(
    ArrayRef2D<std::complex<float>> slc_data_block,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const bool conjugate,
    const std::complex<float> fill_value
);


/**
 * Evaluate and modulate or demodulate the phase carrier onto the given SLC data block.
 * 
 * Note that this function does not perform size checks between `slc_data_block` and the
 * indices blocks and radar grid.
 *
 * @param[out] slc_data_block
 * the block of data SLC to be modulated.
 * unit: array2D of complex
 * @tparam[in] carrier_phase
 * carrier phase of the SLC data, in radians, as a function of azimuth and range.
 * This phase will be modulated to or demodulated from the image.
 * @param[in] radar_grid
 * parameters for the given radar grid corresponding to `slc_data_block`.
 * @param[in] azimuth_indices
 * azimuth index of each pixel in the output block w.r.t the radar grid. Must be the
 * same shape as `slc_data_block`.
 * unit: azimuth row indices (int)
 * @param[in] range_indices
 * range index of each pixel in the output block w.r.t the radar grid. Must be the
 * same shape as `slc_data_block`.
 * unit: range column indices (int)
 * @param[in] conjugate
 * if true, modulate the conjugate of the phase.
 * @param[in] fill_value
 * The value to fill out-of-bounds pixels with. Out-of-bounds pixels are defined here as
 * any pixels that cannot be evaluated by the carrier_phase function. Poly2d functions
 * do not have out-of-bounds pixels, but LUT2d functions may. Defaults to NaN + j*NaN.
 */
template <typename AzRgFunc = isce3::core::Poly2d>
void _modulateAtCoords(
    ArrayRef2D<std::complex<float>> slc_data_block,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const ConstArrayRef2D<double> azimuth_indices,
    const ConstArrayRef2D<double> range_indices,
    const bool conjugate,
    const std::complex<float> fill_value
);

} // namespace isce3::image::modulate
