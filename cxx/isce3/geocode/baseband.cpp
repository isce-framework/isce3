#include "baseband.h"

#include <isce3/core/LUT2d.h>
#include <isce3/core/Matrix.h>

void isce3::geocode::baseband(isce3::core::Matrix<std::complex<float>>& data,
                             const double starting_range,
                             const double sensing_start,
                             const double range_pixel_spacing, const double prf,
                             const isce3::core::LUT2d<double>& doppler_lut)
{

    size_t length = data.length();
    size_t width = data.width();
#pragma omp parallel for
    for (size_t kk = 0; kk < length * width; ++kk) {

        size_t line = kk / width;
        size_t col = kk % width;
        const double azimuth_time = sensing_start + line / prf;
        const double slant_range = starting_range + col * range_pixel_spacing;
        const double phase = doppler_lut.eval(azimuth_time, slant_range) * 2 *
                             M_PI * azimuth_time;
        const std::complex<float> cpx_phase(std::cos(phase), -std::sin(phase));
        data(line, col) *= cpx_phase;
    }
}
