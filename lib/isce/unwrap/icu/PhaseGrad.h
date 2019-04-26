#ifndef ISCE_UNWRAP_ICU_PHASEGRAD_H
#define ISCE_UNWRAP_ICU_PHASEGRAD_H

#include <complex> // std::complex
#include <cstddef> // size_t

namespace isce::unwrap::icu
{

// \brief Compute phase slope in x & y, smoothed by Gaussian kernel.
//
// The algorithm is the same as that developed by Madsen for estimation of the 
// doppler centroid.
//
// @param[out] phasegradx Phase gradient in x
// @param[out] phasegrady Phase gradient in y
// @param[in] intf Interferogram
// @param[in] length Tile length
// @param[in] width Tile width
// @param[in] winsize Kernel size
void calcPhaseGrad(
    float * phasegradx,
    float * phasegrady,
    const std::complex<float> * intf, 
    const size_t length,
    const size_t width,
    const int winsize);

}

#endif /* ISCE_UNWRAP_ICU_PHASEGRAD_H */

