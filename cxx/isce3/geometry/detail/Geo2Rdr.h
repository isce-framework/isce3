#pragma once

#include <isce3/core/forward.h>

#include <isce3/core/Common.h>
#include <isce3/core/LookSide.h>
#include <isce3/error/ErrorCode.h>

namespace isce3 { namespace geometry { namespace detail {

/** \internal Root-finding configuration parameters for geo2rdr */
struct Geo2RdrParams {
    /** \internal Absolute slant range convergence tolerance (m) */
    double threshold = 1e-8;

    /** \internal Maximum number of Newton-Raphson iterations */
    int maxiter = 50;

    /** \internal Step size for computing numerical gradient of Doppler (m) */
    double delta_range = 10.;
};

/**
 * \internal
 * Unified host/device implementation of isce3::geometry::geo2rdr and
 * isce3::cuda::geometry::geo2rdr
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
CUDA_HOSTDEV isce3::error::ErrorCode
geo2rdr(double* t, double* r, const isce3::core::Vec3& llh,
        const isce3::core::Ellipsoid& ellipsoid, const Orbit& orbit,
        const DopplerModel& doppler, double wvl, isce3::core::LookSide side,
        double t0, const Geo2RdrParams& params = {});

}}} // namespace isce3::geometry::detail

#include "Geo2Rdr.icc"
