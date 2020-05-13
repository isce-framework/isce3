#include "baseband.h"

void isce::geocode::baseband(isce::core::Matrix<std::complex<float>> &data,
                    const double startingRange, const double sensingStart,
                    const double rangePixelSpacing, const double prf,
                    const isce::core::LUT2d<double>& dopplerLUT)
{

    size_t length = data.length();
    size_t width = data.width();
    #pragma omp parallel for
    for (size_t kk = 0; kk < length*width; ++kk) {

        size_t line = kk / width;
        size_t col = kk % width;
        const double azTime = sensingStart + line/prf;
        const double rng = startingRange + col * rangePixelSpacing;
        const double phase = dopplerLUT.eval(azTime, rng) * 2*M_PI * azTime;
        //const double phase = dop*az; //modulo_f(dop*az, 2.0*M_PI);
        const std::complex<float> cpxPhase(std::cos(phase), -std::sin(phase));
        data(line,col) *= cpxPhase;

    }
}

