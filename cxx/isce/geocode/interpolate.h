#include <isce/io/forward.h>
#include <isce/core/Matrix.h>
#include <isce/core/Interpolator.h>
#include <iostream>
#include <complex>

namespace isce { namespace geocode {

    void interpolate(isce::core::Matrix<std::complex<float>>& rdrDataBlock,
           isce::core::Matrix<std::complex<float>>& geoDataBlock,
             const std::valarray<double>& radarX, const std::valarray<double>& radarY,
             const std::valarray<std::complex<double>> geometricalPhase,
             const int radarBlockWidth, const int radarBlockLength,
             const int azimuthFirstLine, const int rangeFirstPixel,
             isce::core::Interpolator<std::complex<float>> * interp);

}
}
