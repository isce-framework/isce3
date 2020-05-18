#include <isce/io/forward.h>
#include <isce/core/Matrix.h>
#include <isce/core/Interpolator.h>
#include <iostream>
#include <complex>

namespace isce { namespace geocode {

    /**
     * @param[in] rdrDataBlock a block of data in radar coordinate
     * @param[in] geoDataBlock a block of data in geo coordinates
     * @param[in] radarX the radar-coordinates x-index of the pixels in geo-grid
     * @param[in] radarY the radar-coordinates y-index of the pixels in geo-grid 
     * @param[in] geometricalPhase the geometrical phase of each pixel in geo-hrid to be removed from the gocoded data after interpolation 
     * @param[in] radarBlockWidth width of the data block in radar coordinates
     * @param[in] radarBlockLength length of the data block in radar coordinates
     * @param[in] azimuthFirstLine azimuth time of the first sample
     * @param[in] rangeFirstPixel  range of the first sample
     * @param[in] interp interpolator object
     */
        //const double phase = dop*az; //modulo_f(dop*az, 2.0*M_PI);
    void interpolate(isce::core::Matrix<std::complex<float>>& rdrDataBlock,
           isce::core::Matrix<std::complex<float>>& geoDataBlock,
             const std::valarray<double>& radarX, const std::valarray<double>& radarY,
             const std::valarray<std::complex<double>> geometricalPhase,
             const int radarBlockWidth, const int radarBlockLength,
             const int azimuthFirstLine, const int rangeFirstPixel,
             isce::core::Interpolator<std::complex<float>> * interp);

}
}
