#pragma once

#include "forward.h"
#include <isce3/core/forward.h>

#include <cmath>

#include <isce3/geometry/detail/Rdr2Geo.h>

namespace isce3 { namespace geometry {

/**
 * Transform from radar coordinates (range, azimuth) to ECEF XYZ
 * coordinates.
 *
 * \param[in]  aztime     Target azimuth time w.r.t. orbit reference epoch (s)
 * \param[in]  slantRange Target slant range (m)
 * \param[in]  doppler    Doppler centroid at target position (Hz)
 * \param[in]  orbit      Platform orbit
 * \param[in]  dem        DEM sampling interface
 * \param[out] targetXYZ  Output target ECEF XYZ position (m)
 * \param[in]  wavelength Radar wavelength (wrt requested Doppler) (m)
 * \param[in]  side       Radar look side
 * \param[in]  tolHeight  Allowable height error of solution (m)
 * \param[in]  lookMin    Smallest possible pseudo-look angle (rad)
 * \param[in]  lookMax    Largest possible pseudo-look angle (rad)
 *
 * Note: Usually the look angle is defined as the angle between the line of
 * sight vector and the nadir vector.  Here the pseudo-look angle is defined in
 * a similar way, except it's with respect to the projection of the nadir vector
 * into a plane perpendicular to the velocity.  For simplicity, we use the
 * geocentric nadir definition.
 */
int rdr2geo_bracket(double aztime, double slantRange, double doppler,
        const isce3::core::Orbit& orbit,
        const isce3::geometry::DEMInterpolator& dem,
        isce3::core::Vec3& targetXYZ, double wavelength,
        isce3::core::LookSide side,
        double tolHeight = isce3::geometry::detail::DEFAULT_TOL_HEIGHT,
        double lookMin = 0.0, double lookMax = M_PI / 2);

}} // namespace isce3::geometry
