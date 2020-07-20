#include "interpolate.h"

void isce::geocode::interpolate(
        const isce::core::Matrix<std::complex<float>>& rdrDataBlock,
        isce::core::Matrix<std::complex<float>>& geoDataBlock,
        const std::valarray<double>& radarX,
        const std::valarray<double>& radarY,
        const std::valarray<std::complex<double>>& geometricalPhase,
        const int radarBlockWidth, const int radarBlockLength,
        const int azimuthFirstLine, const int rangeFirstPixel,
        const isce::core::Interpolator<std::complex<float>>* interp)
{

    size_t length = geoDataBlock.length();
    size_t width = geoDataBlock.width();
    int extraMargin = isce::core::SINC_HALF;

#pragma omp parallel for
    for (size_t kk = 0; kk < length * width; ++kk) {

        size_t i = kk / width;
        size_t j = kk % width;

        // adjust the row and column indicies for the current block,
        // i.e., moving the origin to the top-left of this radar block.
        double rdrY = radarY[i * width + j] - azimuthFirstLine;
        double rdrX = radarX[i * width + j] - rangeFirstPixel;

        if (rdrX < extraMargin || rdrY < extraMargin ||
            rdrX >= (radarBlockWidth - extraMargin) ||
            rdrY >= (radarBlockLength - extraMargin)) {

            geoDataBlock(i,j) = std::complex<float> (0.0, 0.0);

        } else {

            // Interpolate chip
            const std::complex<double> cval =
                interp->interpolate(rdrX, rdrY, rdrDataBlock);

            // geometricalPhase is the sum of carrier (Doppler) phase to be added
            // back and the geometrical phase to be removed: exp(1J* (carrier
            // - 4.0*PI*slantRange/wavelength))
            geoDataBlock(i, j) = cval * geometricalPhase[i * width + j];
        }
    } // end for
}
