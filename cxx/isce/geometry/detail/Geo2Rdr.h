#pragma once

#include <isce/core/forward.h>

#include <isce/core/Common.h>
#include <isce/core/LookSide.h>
#include <isce/error/ErrorCode.h>

namespace isce { namespace geometry { namespace detail {

/** \internal Root-finding configuration parameters for geo2rdr */
struct Geo2RdrParams {
    /** \internal Absolute slant range convergence tolerance (m) */
    double tol = 1e-8;

    /** \internal Maximum number of Newton-Raphson iterations */
    int maxiter = 50;

    /** \internal Step size for computing numerical gradient of Doppler (m) */
    double dr = 10.;
};

/**
 * \internal
 * Unified host/device implementation of isce::geometry::geo2rdr and
 * isce::cuda::geometry::geo2rdr
 *
 * Transform from geodetic coordinates (longitude, latitude, height) to radar
 * coordinates (azimuth, range).
 *
 * The behavior is undefined if either \p t or \p r is \p NULL
 *
 * \param[out] t         Target azimuth time w.r.t. orbit reference epoch (s)
 * \param[out] r         Target slant range (m)
 * \param[in]  llh       Target lon/lat/hae (deg/deg/m)
 * \param[in]  ellipsoid Reference ellipsoid
 * \param[in]  orbit     Platform orbit
 * \param[in]  doppler   Doppler model as a function of azimuth & range (Hz)
 * \param[in]  wvl       Radar wavelength (m)
 * \param[in]  side      Radar look side
 * \param[in]  t0        Initial azimuth time guess (s)
 * \param[in]  params    Root-finding algorithm parameters
 */
template<class Orbit, class DopplerModel>
CUDA_HOSTDEV isce::error::ErrorCode
geo2rdr(double* t, double* r, const isce::core::Vec3& llh,
        const isce::core::Ellipsoid& ellipsoid, const Orbit& orbit,
        const DopplerModel& doppler, double wvl, isce::core::LookSide side,
        double t0, const Geo2RdrParams& params = {});

}}} // namespace isce::geometry::detail

#include "Geo2Rdr.icc"
