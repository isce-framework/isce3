#include "Modulate.h"

#include <isce3/core/Constants.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Poly2d.h>

namespace isce3::image::modulate {


template<typename AzRgFunc = isce3::core::Poly2d>
std::complex<double> _getPixelCarrierPhase(
    const double azimuth,
    const double range,
    const AzRgFunc& carrier_phase,
    bool conjugate = false
) {
      // Evaluate the pixel's carrier phase
      // unit: phase (radians)
      const double phase = carrier_phase.eval(azimuth, range);

      // Convert the carrier phase into a unit phasor (i.e. an angle on the unit
      // circle in the complex plane).
      // unit: unitless (complex)
      if(conjugate){
          return std::complex<double>(std::cos(phase), -std::sin(phase));
      } else {
          return std::complex<double>(std::cos(phase), std::sin(phase));
      }
}


template<typename AzRgFunc>
void getCarrierPhase(
    ArrayRef2D<std::complex<float>> out,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const bool conjugate,
    const std::complex<float> fill_value
)
{
    // unit: azimuth row indices (int)
    const size_t phase_block_length = out.rows();
    // unit: range column indices (int)
    const size_t phase_block_width = out.cols();

    // remove carrier from radar data
#pragma omp parallel for collapse(2)
    for (size_t az_index = 0; az_index < phase_block_length; ++az_index) {
        for (size_t rg_index = 0; rg_index < phase_block_width; ++rg_index) {
            // unit: time (seconds)
            const double azimuth =
                radar_grid.sensingStart() + az_index / radar_grid.prf();

            // unit: distance (meters)
            const double range = radar_grid.startingRange() +
                    rg_index * radar_grid.rangePixelSpacing();

            if(not carrier_phase.contains(azimuth, range)){
                out(az_index, rg_index) = fill_value;
                continue;
            }
            
            // Get the carrier unit phasor for this pixel
            const auto phasor = _getPixelCarrierPhase(
                azimuth, range, carrier_phase, conjugate
            );
            
            // Write the phasor into the output data block
            out(az_index, rg_index) = phasor;
        }
    } // end multithreaded block
}


template<typename AzRgFunc>
void _getCarrierPhaseAtCoords(
    ArrayRef2D<std::complex<float>> out,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const ConstArrayRef2D<double> azimuth_indices,
    const ConstArrayRef2D<double> range_indices,
    const bool conjugate,
    const std::complex<float> fill_value
)
{
    const size_t outWidth = out.cols();
    const size_t outLength = out.rows();

#pragma omp parallel for collapse(2)
    for (size_t az_index = 0; az_index < outLength; ++az_index){
        for (size_t rg_index = 0; rg_index < outWidth; ++rg_index){

            // Get the indices on the output grid corresponding to these indices
            // on the input grid overall.
            const double az_carrier_index = azimuth_indices(az_index, rg_index);
            const double rg_carrier_index = range_indices(az_index, rg_index);

            // Azimuth time at the current output pixel
            const double azimuth = radar_grid.sensingStart() + az_carrier_index / 
                radar_grid.prf();

            // Slant Range at the current output pixel
            const double range = radar_grid.startingRange() + rg_carrier_index *
                radar_grid.rangePixelSpacing();

            if(not carrier_phase.contains(azimuth, range)){
                out(az_index, rg_index) = fill_value;
                continue;
            }
            
            // Get the carrier phasor for this pixel
            const auto phasor = _getPixelCarrierPhase(
                azimuth, range, carrier_phase, conjugate
            );
            
            // Write the phasor into the output data block
            out(az_index, rg_index) = phasor;
        }
    } // end multithreaded block
}


template<typename AzRgFunc>
void modulateCarrierPhase(
    ArrayRef2D<std::complex<float>> slc_data_block,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const bool conjugate,
    const std::complex<float> fill_value
)
{
    // unit: azimuth row indices (int)
    const size_t phase_block_length = slc_data_block.rows();
    // unit: range column indices (int)
    const size_t phase_block_width = slc_data_block.cols();

    // remove carrier from radar data
#pragma omp parallel for collapse(2)
    for ( size_t az_index = 0; az_index < phase_block_length; ++az_index) {
        for (size_t rg_index = 0; rg_index < phase_block_width; ++rg_index) {
            // unit: time (seconds)
            const double azimuth =
                radar_grid.sensingStart() + az_index / radar_grid.prf();

            // unit: distance (meters)
            const double range = radar_grid.startingRange() +
                    rg_index * radar_grid.rangePixelSpacing();

            if(not carrier_phase.contains(azimuth, range)){
                slc_data_block(az_index, rg_index) = fill_value;
                continue;
            }
            
            // Get the carrier phasor for this pixel
            const auto phasor = _getPixelCarrierPhase(
                azimuth, range, carrier_phase, conjugate
            );
            
            // Modulate the phasor into the output data block
            slc_data_block(az_index, rg_index) *= phasor;
        }
    } // end multithreaded block
}


template<typename AzRgFunc>
void _modulateCarrierPhaseAtCoords(
    ArrayRef2D<std::complex<float>> slc_data_block,
    const AzRgFunc& carrier_phase,
    const isce3::product::RadarGridParameters& radar_grid,
    const ConstArrayRef2D<double> azimuth_indices,
    const ConstArrayRef2D<double> range_indices,
    const bool conjugate,
    const std::complex<float> fill_value
)
{
    const size_t outWidth = slc_data_block.cols();
    const size_t outLength = slc_data_block.rows();

#pragma omp parallel for collapse(2)
    for (size_t az_index = 0; az_index < outLength; ++az_index){
        for (size_t rg_index = 0; rg_index < outWidth; ++rg_index){

            // Get the indices on the output grid corresponding to these indices
            // on the input grid overall.
            const double az_carrier_index = azimuth_indices(az_index, rg_index);
            const double rg_carrier_index = range_indices(az_index, rg_index);

            // Azimuth time at the current output pixel
            const double azimuth = radar_grid.sensingStart() + az_carrier_index / 
                radar_grid.prf();

            // Slant Range at the current output pixel
            const double range = radar_grid.startingRange() + rg_carrier_index *
                radar_grid.rangePixelSpacing();

            if(not carrier_phase.contains(azimuth, range)){
                slc_data_block(az_index, rg_index) = fill_value;
                continue;
            }

            // Get the carrier phasor for this pixel
            const auto phasor = _getPixelCarrierPhase(
                azimuth, range, carrier_phase, conjugate
            );
            
            // Modulate the phasor into the output data block
            slc_data_block(az_index, rg_index) *= phasor;
        }
    } // end multithreaded block
}


#define EXPLICIT_INSTANTIATION(AzRgFunc)                                      \
template void getCarrierPhase(                                                \
    ArrayRef2D<std::complex<float>> out,                                      \
    const AzRgFunc& carrier_phase,                                            \
    const isce3::product::RadarGridParameters& radar_grid,                    \
    const bool conjugate,                                                     \
    const std::complex<float> fill_value                                      \
);                                                                            \
template void _getCarrierPhaseAtCoords(                                       \
    ArrayRef2D<std::complex<float>> out,                                      \
    const AzRgFunc& carrier_phase,                                            \
    const isce3::product::RadarGridParameters& radar_grid,                    \
    const ConstArrayRef2D<double> azimuth_indices,                            \
    const ConstArrayRef2D<double> range_indices,                              \
    const bool conjugate,                                                     \
    const std::complex<float> fill_value                                      \
);                                                                            \
template void modulateCarrierPhase(                                           \
    ArrayRef2D<std::complex<float>> slc_data_block,                           \
    const AzRgFunc& carrier_phase,                                            \
    const isce3::product::RadarGridParameters& radar_grid,                    \
    const bool conjugate,                                                     \
    const std::complex<float> fill_value                                      \
);                                                                            \
template void _modulateCarrierPhaseAtCoords(                                  \
    ArrayRef2D<std::complex<float>> slc_data_block,                           \
    const AzRgFunc& carrier_phase,                                            \
    const isce3::product::RadarGridParameters& radar_grid,                    \
    const ConstArrayRef2D<double> azimuth_indices,                            \
    const ConstArrayRef2D<double> range_indices,                              \
    const bool conjugate,                                                     \
    const std::complex<float> fill_value                                      \
)
EXPLICIT_INSTANTIATION(isce3::core::LUT2d<double>);
EXPLICIT_INSTANTIATION(isce3::core::Poly2d);

} // end namespace isce3::image::modulate
