#include "Resample.h"

#include <isce3/core/Constants.h>
#include <isce3/core/Interpolator.h>
#include <isce3/core/LUT2d.h>

namespace isce3::image::v2 {


void resampleToCoords(
    ArrayRef2D<std::complex<float>> resampled_data_block,
    const ConstArrayRef2D<std::complex<float>> input_data_block,
    const ConstArrayRef2D<double> range_input_indices,
    const ConstArrayRef2D<double> azimuth_input_indices,
    const isce3::product::RadarGridParameters& radar_grid,
    const isce3::core::LUT2d<double>& native_doppler_lut,
    const std::complex<float> fill_value
)
{
    // Instantiate the interpolator.
    // A small change over previous versions - a pointer to this object would previously
    // been passed as an argument to this function. In order to make this code directly
    // callable at the Python level, this has been moved here. In order to add support
    // for different interpolators, a Python binding needs to be made for these
    // interpolator objects.
    auto sinc_interp =
        isce3::core::Sinc2dInterpolator<std::complex<float>>(
        isce3::core::SINC_LEN, isce3::core::SINC_SUB);

    // number of columns on input array
    const int in_width = input_data_block.cols();
    // number of rows on input array
    const int in_length = input_data_block.rows();
    // number of columns on output array
    const int out_width = resampled_data_block.cols();
    // number of rows on output array
    const int out_length = resampled_data_block.rows();

    // Sinc interpolation chip dimensions:
    // unit: number of pixels (int)
    const int chip_size = isce3::core::SINC_ONE;
    // unit: number of pixels (int)
    const int chip_half = isce3::core::SINC_HALF;

    // Parallel block
    #pragma omp parallel
    {
        // Allocate matrix for working sinc chip
        isce3::core::Matrix<std::complex<float>> chip(chip_size, chip_size);

        // Enter the parallel loop.
        #pragma omp for collapse(2)
        for (size_t az_resamp_ind = 0; az_resamp_ind < out_length; ++az_resamp_ind) {
            for (size_t rg_resamp_ind = 0; rg_resamp_ind < out_width; ++rg_resamp_ind) {
                // Populate the position with the fill value first.
                // Having this happen first means that the `continue` statement can
                // be used any time a pixel cannot be interpolated.
                resampled_data_block(az_resamp_ind, rg_resamp_ind) = fill_value;

                // The indices on the resampled data block
                // unit: column pixels on input array (double)
                const auto range_input_ind =
                    range_input_indices(az_resamp_ind, rg_resamp_ind);
                // unit: row pixels on input array (double)
                const auto azimuth_input_ind =
                    azimuth_input_indices(az_resamp_ind, rg_resamp_ind);

                // Skip if either the azimuth or range input index are NaN.
                if (std::isnan(azimuth_input_ind) || std::isnan(range_input_ind)) {
                    continue;
                }

                // unit: range column indices (int)
                const auto range_input_ind_int = static_cast<int>(range_input_ind);
                // unit: azimuth row indices (int)
                const auto azimuth_input_ind_int = static_cast<int>(azimuth_input_ind);
                
                // Check if chip indices could be outside radar grid minus margin to
                // account for sinc chip. Fill with fill_value and skip if chip indices
                // out of bounds.
                if ((range_input_ind_int < chip_half) ||
                    (range_input_ind_int >= (in_width - chip_half)))
                    continue;
                if ((azimuth_input_ind_int < chip_half) ||
                    (azimuth_input_ind_int >= (in_length - chip_half)))
                    continue;

                // unit: range column indices (double)
                const auto range_input_index_remainder =
                    range_input_ind - static_cast<double>(range_input_ind_int);
                // unit: azimuth row indices (double)
                const auto azimuth_input_index_remainder =
                    azimuth_input_ind - static_cast<double>(azimuth_input_ind_int);

                // Slant Range at the current output pixel
                // unit: distance (meters)
                const double rg_distance =
                    radar_grid.startingRange() +
                    range_input_ind *
                    radar_grid.rangePixelSpacing();

                // Azimuth time at the current output pixel
                // unit: time (seconds)
                const double az_time =
                    radar_grid.sensingStart() +
                    azimuth_input_ind /
                    radar_grid.prf();

                // If the doppler LUT doesn't contain this coordinate, fill this pixel
                // with the given fill_value and skip it.
                if (not native_doppler_lut.contains(az_time, rg_distance))
                    continue;
                
                // Evaluate doppler at current range and azimuth time
                // unit: frequency (radians per sample)
                const double doppler_freq =
                    native_doppler_lut.eval(az_time, rg_distance) * 2 * M_PI / 
                    radar_grid.prf();

                // Read data chip
                for (int i_chip = 0; i_chip < chip_size; ++i_chip) {
                    // Row to read from in Azimuth coordinates
                    // unit: azimuth row indices (int)
                    const int az_chip_ind = azimuth_input_ind_int + i_chip - chip_half;

                    // Doppler phase at chip pixel
                    // unit: phase (radians)
                    const double doppler_phase = doppler_freq * (i_chip - chip_half);

                    // Compute doppler phase to be removed from radar data.
                    // (i.e. as a unit vector on the complex plane.)
                    const std::complex<float> doppler_phase_conj(
                        std::cos(doppler_phase), -std::sin(doppler_phase));
                    
                    // Set the data values after removing doppler in azimuth
                    for (int j_chip = 0; j_chip < chip_size; ++j_chip) {
                        // Column to read from in Range coordinates
                        // unit: range column indices (int)
                        const int rg_chip_ind = 
                            range_input_ind_int + j_chip - chip_half;

                        // Set the point at the chip indices to their value on the data
                        // block, rotated by the doppler conjugate phasor.
                        chip(i_chip, j_chip) = 
                            input_data_block(az_chip_ind, rg_chip_ind) * 
                            doppler_phase_conj;
                    }
                }

                // Interpolation performed on data stripped of doppler.
                // Calculate the doppler shift doppler shift to be reintroduced.
                const double doppler_resampled_phase =
                    doppler_freq * azimuth_input_index_remainder;
                const std::complex<float> doppler_resampled_phasor(
                    std::cos(doppler_resampled_phase),
                    std::sin(doppler_resampled_phase)
                );

                // Interpolate chip
                const std::complex<float> interpolated_complex_val =
                    sinc_interp.interpolate(
                        chip_half + range_input_index_remainder,
                        chip_half + azimuth_input_index_remainder,
                        chip
                    );

                // Add doppler to interpolated value
                resampled_data_block(az_resamp_ind, rg_resamp_ind) = 
                    interpolated_complex_val * doppler_resampled_phasor;

            }
        
        } // end omp for collapse(2)
    } // end omp parallel
} // end resampleToCoords

} // end namespace isce3::image::v2
