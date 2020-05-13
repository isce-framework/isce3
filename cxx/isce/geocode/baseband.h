#pragma once
#include <isce/io/forward.h>
#include <isce/core/forward.h>
#include <isce/core/Matrix.h>
#include <isce/core/LUT2d.h>
#include <iostream>
#include <complex>

namespace isce { namespace geocode {

    void baseband(isce::core::Matrix<std::complex<float>> &data,
              const double startingRange, const double sensingStart,
                const double rangePixelSpacing, const double prf,
                const isce::core::LUT2d<double>& dopplerLUT);


}
}
