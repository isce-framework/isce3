#pragma once

#include <isce3/core/forward.h>

#include <isce3/geometry/detail/Geo2Rdr.h>

namespace isce3 { namespace geometry {

/** Solve the inverse mapping problem using a derivative-free method.
 *
 * @param[in]    x          Target position, XYZ in m
 * @param[in]    orbit      Platform position vs time
 * @param[in]    doppler    Geometric Doppler centroid as function of
 *                          (time, range), Hz
 * @param[out]   aztime     Time when Doppler centroid crosses target, s
 * @param[out]   range      Distance to target at aztime, m
 * @param[in]    wavelength Radar wavelength used to scale Doppler, m
 * @param[in]    side       Radar look direction, Left or Right
 * @param[in]    tolAzTime  Azimuth convergence tolerance, s
 * @param[in]    timeStart  Start of search interval, s
 *                          Defaults to max of orbit and Doppler LUT start time
 * @param[in]    timeEnd    End of search interval, s
 *                          Defaults to min of orbit and Doppler LUT end time
 *
 * @returns Nonzero when successfully converged, zero if iterations exceeded.
 * aztime and range are always updated to the best available estimate.
 */
int geo2rdr_bracket(const isce3::core::Vec3& x, const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& doppler, double& aztime,
        double& range, double wavelength, isce3::core::LookSide side,
        double tolAzTime = isce3::geometry::detail::DEFAULT_TOL_AZ_TIME,
        std::optional<double> timeStart = std::nullopt,
        std::optional<double> timeEnd = std::nullopt);

}} // namespace isce3::geometry
