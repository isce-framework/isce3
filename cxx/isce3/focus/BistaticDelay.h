#pragma once

#include <isce3/core/Common.h>
#include <isce3/core/forward.h>

namespace isce3 { namespace focus {

/**
 * Compute the two-way propagation delay between the radar antenna phase center
 * and a target scatterer.
 *
 * The delay model applies a bistatic correction for the displacement of the
 * radar between transmit and receive time.
 *
 * Speed of light in vacuum is assumed. No corrections are applied for delays
 * due to atmospheric effects.
 *
 * \param[in] p Antenna phase center position (m) at transmit time
 * \param[in] v Antenna phase center velocity (m/s) at transmit time
 * \param[in] x Target position (m)
 * \returns     The bistatic propagation delay (s)
 */
CUDA_HOSTDEV
double bistaticDelay(const isce3::core::Vec3 & p,
                     const isce3::core::Vec3 & v,
                     const isce3::core::Vec3 & x);

}}

#include "BistaticDelay.icc"
