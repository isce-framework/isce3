#include "interpolate.h"

void isce3::geocode::interpolate(
        const isce3::core::Matrix<std::complex<float>>& rdrDataBlock,
        isce3::core::Matrix<std::complex<float>>& geoDataBlock,
        const std::valarray<double>& radarX,
        const std::valarray<double>& radarY, const int azimuthFirstLine,
        const int rangeFirstPixel,
        const isce3::core::Interpolator<std::complex<float>>* interp,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::LUT2d<double>& dopplerLUT, const bool& flatten)
{
    const int chipSize = isce3::core::SINC_ONE;
    const int width = geoDataBlock.width();
    const int length = geoDataBlock.length();
    const int inWidth = rdrDataBlock.width();
    const int inLength = rdrDataBlock.length();
    const int chipHalf = chipSize / 2;

#pragma omp parallel for
    for (size_t kk = 0; kk < length * width; ++kk) {
        size_t i = kk / width;
        size_t j = kk % width;

        // adjust the row and column indicies for the current block,
        // i.e., moving the origin to the top-left of this radar block.
        double rdrY = radarY[i * width + j] - azimuthFirstLine;
        double rdrX = radarX[i * width + j] - rangeFirstPixel;

        const int intX = static_cast<int>(rdrX);
        const int intY = static_cast<int>(rdrY);
        const double fracX = rdrX - intX;
        const double fracY = rdrY - intY;

        if ((intX < chipHalf) || (intX >= (inWidth - chipHalf))) {
            geoDataBlock(i, j) *= std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        if ((intY < chipHalf) || (intY >= (inLength - chipHalf))) {
            geoDataBlock(i, j) *= std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        // Slant Range at the current output pixel
        const double rng =
                radarGrid.startingRange() +
                radarX[i * width + j] * radarGrid.rangePixelSpacing();

        // Azimuth time at the current output pixel
        const double az = radarGrid.sensingStart() +
                          radarY[i * width + j] / radarGrid.prf();

        if (not dopplerLUT.contains(az, rng)) {
            geoDataBlock(i, j) *= std::numeric_limits<float>::quiet_NaN();
            continue;
        }

        // Evaluate Doppler at current range and azimuth time
        const double dop =
                dopplerLUT.eval(az, rng) * 2 * M_PI / radarGrid.prf();

        // Doppler to be added back. Simultaneously evaluate carrier
        // that needs to be added back after interpolation
        double carrier_phase = (dop * fracY); // +
                                              //_rgCarrier.eval(rdrX, rdrY) +
                                              //_azCarrier.eval(rdrX, rdrY);

        if (flatten) {
            carrier_phase += (4.0 * (M_PI / radarGrid.wavelength())) * rng;
        }

        isce3::core::Matrix<std::complex<float>> chip(chipSize, chipSize);
        // Read data chip without the carrier phases
        for (int ii = 0; ii < chipSize; ++ii) {
            // Row to read from
            const int chipRow = intY + ii - chipHalf;

            // Carrier phase
            const double phase = dop * (ii - chipHalf);
            const std::complex<float> cval(std::cos(phase), -std::sin(phase));

            // Set the data values after removing doppler in azimuth
            for (int jj = 0; jj < chipSize; ++jj) {
                // Column to read from
                const int chipCol = intX + jj - chipHalf;
                chip(ii, jj) = rdrDataBlock(chipRow, chipCol) * cval;
            }
        }
        // Interpolate chip
        const std::complex<float> cval =
                interp->interpolate(isce3::core::SINC_HALF + fracX,
                        isce3::core::SINC_HALF + fracY, chip);

        geoDataBlock(i, j) = cval * std::complex<float>(std::cos(carrier_phase),
                                            std::sin(carrier_phase));
    }
}
