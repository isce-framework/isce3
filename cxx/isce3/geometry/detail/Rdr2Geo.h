#pragma once

#include <isce3/core/forward.h>

#include <isce3/core/Common.h>
#include <isce3/core/LookSide.h>
#include <isce3/error/ErrorCode.h>

namespace isce { namespace geometry { namespace detail {

/** \internal Root-finding configuration parameters for rdr2geo */
struct Rdr2GeoParams {
    /** \internal Absolute slant range convergence tolerance (m) */
    double tol = 1e-8;

    /** \internal Maximum number of primary Newton-Raphson iterations */
    int maxiter = 25;

    /** \internal Maximum number of secondary iterations */
    int extraiter = 15;
};

/**
 * \internal
 * Unified host/device implementation of isce::geometry::rdr2geo and
 * isce::cuda::geometry::rdr2geo
 *
 * Transform from radar coordinates (azimuth, range) to geodetic coordinates
 * (longitude, latitude, height).
 *
 * The behavior is undefined if \p llh is \p NULL
 *
 * \param[out] llh       Output target lon/lat/hae (deg/deg/m)
 * \param[in]  t         Target azimuth time w.r.t. orbit reference epoch (s)
 * \param[in]  r         Target slant range (m)
 * \param[in]  fd        Doppler centroid at target position (Hz)
 * \param[in]  orbit     Platform orbit
 * \param[in]  dem       DEM sampling interface
 * \param[in]  ellipsoid DEM reference ellipsoid
 * \param[in]  wvl       Radar wavelength (m)
 * \param[in]  side      Radar look side
 * \param[in]  h0        Initial target height estimate (m)
 * \param[in]  params    Root-finding algorithm parameters
 */
template<class Orbit, class DEMInterpolator>
CUDA_HOSTDEV isce::error::ErrorCode
rdr2geo(isce::core::Vec3* llh, double t, double r, double fd,
        const Orbit& orbit, const DEMInterpolator& dem,
        const isce::core::Ellipsoid& ellipsoid, double wvl,
        isce::core::LookSide side, double h0 = 0.,
        const Rdr2GeoParams& params = {});

/**
 * \internal
 * Implementation of isce::geometry::rdr2geo
 *
 * Transform from radar coordinates (range, azimuth) to geodetic coordinates
 * (longitude, latitude, height).
 *
 * The behavior is undefined if \p llh is \p NULL
 *
 * \param[out] llh       Output target lon/lat/hae (deg/deg/m)
 * \param[in]  pixel     Target pixel
 * \param[in]  tcnbasis  Geocentric TCN basis corresponding to pixel
 * \param[in]  pos       Platform position vector
 * \param[in]  vel       Platform velocity vector
 * \param[in]  dem       DEM sampling interface
 * \param[in]  ellipsoid DEM reference ellipsoid
 * \param[in]  side      Radar look side
 * \param[in]  h0        Initial target height estimate (m)
 * \param[in]  params    Root-finding algorithm parameters
 */
template<class DEMInterpolator>
CUDA_HOSTDEV isce::error::ErrorCode
rdr2geo(isce::core::Vec3* llh, const isce::core::Pixel& pixel,
        const isce::core::Basis& tcnbasis, const isce::core::Vec3& pos,
        const isce::core::Vec3& vel, const DEMInterpolator& dem,
        const isce::core::Ellipsoid& ellipsoid, isce::core::LookSide side,
        double h0 = 0., const Rdr2GeoParams& params = {});

}}} // namespace isce::geometry::detail

#include "Rdr2Geo.icc"
