#pragma once
#include <isce/io/forward.h>
#include <isce/core/forward.h>
#include <isce/core/Matrix.h>
#include <isce/core/LUT2d.h>
#include <iostream>
#include <complex>

namespace isce { namespace geocode {
    /**
     * param[in] data a matrix of data that needs to be sub-banded in azimuth direction
     * param[in] startingRange starting range of the data block
     * param[in] sensingStart  starting azimuth time of the data block
     * param[in] rangePixelSpacing spacing of the slant range
     * param[in] prf pulse repetition frequency
     * param[in] dopplerLUT 2D LUT of the image Doppler
     */
    void baseband(isce::core::Matrix<std::complex<float>> &data,
              const double startingRange, const double sensingStart,
                const double rangePixelSpacing, const double prf,
                const isce::core::LUT2d<double>& dopplerLUT);


}
}
