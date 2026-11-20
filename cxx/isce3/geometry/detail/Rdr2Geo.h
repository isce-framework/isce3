#pragma once

#include <isce3/core/forward.h>

#include <cmath>

#include <isce3/core/Common.h>
#include <isce3/core/LookSide.h>
#include <isce3/error/ErrorCode.h>

namespace isce3 { namespace geometry { namespace detail {

/** \internal Root-finding configuration parameters for rdr2geo */
struct Rdr2GeoParams {
    /** \internal Absolute slant range convergence tolerance (m) */
    double threshold = 1e-8;

    /** \internal Maximum number of primary Newton-Raphson iterations */
    int maxiter = 25;

    /** \internal Maximum number of secondary iterations */
    int extraiter = 15;
};

/**
 * \internal
 * Unified host/device implementation of isce3::geometry::rdr2geo and
 * isce3::cuda::geometry::rdr2geo
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
CUDA_HOSTDEV isce3::error::ErrorCode
rdr2geo(isce3::core::Vec3* llh, double t, double r, double fd,
        const Orbit& orbit, const DEMInterpolator& dem,
        const isce3::core::Ellipsoid& ellipsoid, double wvl,
        isce3::core::LookSide side, double h0 = 0.,
        const Rdr2GeoParams& params = {});

/**
 * \internal
 * Implementation of isce3::geometry::rdr2geo
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
CUDA_HOSTDEV isce3::error::ErrorCode
rdr2geo(isce3::core::Vec3* llh, const isce3::core::Pixel& pixel,
        const isce3::core::Basis& tcnbasis, const isce3::core::Vec3& pos,
        const isce3::core::Vec3& vel, const DEMInterpolator& dem,
        const isce3::core::Ellipsoid& ellipsoid, isce3::core::LookSide side,
        double h0 = 0., const Rdr2GeoParams& params = {});


/** Default convergence tolerance for height (meters) */
inline constexpr double DEFAULT_TOL_HEIGHT = 1e-5;

/** \internal Root-finding configuration parameters for rdr2geo_bracket */
struct Rdr2GeoBracketParams {
    /** \internal Allowable height error of solution (m) */
    double tol_height = DEFAULT_TOL_HEIGHT;

    /** \internal Smallest possible pseudo-look angle (rad) */
    double look_min = 0.0;

    /** \internal Largest possible pseudo-look angle (rad) */
    double look_max = M_PI / 2;
};

/**
 * \internal
 * Implementation of isce3::geometry::rdr2geo_bracket
 *
 * Transform from radar coordinates (range, azimuth) to ECEF XYZ
 * coordinates.
 *
 * The behavior is undefined if \p xyz is \p NULL
 *
 * \param[out] xyz        Output target ECEF XYZ position (m)
 * \param[in]  aztime     Target azimuth time w.r.t. orbit reference epoch (s)
 * \param[in]  slantRange Target slant range (m)
 * \param[in]  doppler    Doppler centroid at target position (Hz)
 * \param[in]  orbit      Platform orbit
 * \param[in]  dem        DEM sampling interface
 * \param[in]  ellipsoid  DEM reference ellipsoid
 * \param[in]  wavelength Radar wavelength (wrt requested Doppler) (m)
 * \param[in]  side       Radar look side
 * \param[in]  params     Root finding algorithm parameters
 *
 * Note: Usually the look angle is defined as the angle between the line of
 * sight vector and the nadir vector.  Here the pseudo-look angle is defined in
 * a similar way, except it's with respect to the projection of the nadir vector
 * into a plane perpendicular to the velocity.  For simplicity, we use the
 * geocentric nadir definition.
 */
template<class Orbit, class DEMInterpolator>
CUDA_HOSTDEV isce3::error::ErrorCode
rdr2geo_bracket(isce3::core::Vec3* xyz,
        double aztime, double slantRange, double doppler, const Orbit& orbit,
        const DEMInterpolator& dem, const isce3::core::Ellipsoid& ellipsoid,
        double wavelength, isce3::core::LookSide side,
        const Rdr2GeoBracketParams& params = {});


/**
 * \internal
 * Lower level version of rdr2geo_bracket that works directly in polar
 * coordinates.  Also avoids repeated Orbit interpolations and trig calls.
 *
 * @param[out] xyz              Output target ECEF XYZ position (m)
 * @param[out] lookAngle        Output pseudo-look angle of target (rad)
 * @param[in]  origin           Origin of the polar grid, ECEF XYZ (m)
 * @param[in]  axis             Along-track axis of polar grid, unit ECEF XYZ
 * @param[in]  slantRange       Distance from origin to target (m)
 * @param[in]  sinSquint        Sine of squint angle
 * @param[in]  cosSquint        Cosine of squint angle
 * @param[in]  dem              Digital elevation model (m above ellipsoid)
 * @param[in]  ellipsoid        Ellipsoid associated with DEM
 * @param[in]  side             Look direction (Left or Right)
 * @param[in]  params           Root finding algorithm parameters
 */
template<class DEMInterpolator>
CUDA_HOSTDEV isce3::error::ErrorCode
polar2geo_bracket(isce3::core::Vec3* xyz, double* lookAngle,
        const isce3::core::Vec3& origin, const isce3::core::Vec3& axis,
        const double slantRange, const double sinSquint, const double cosSquint,
        const DEMInterpolator& dem, const isce3::core::Ellipsoid& ellipsoid,
        isce3::core::LookSide side, const Rdr2GeoBracketParams& params);


/**
 * \internal
 * Low level version of polar2polar_bracket.
 *
 * @param[in]  outSinSquint     Sine of squint angle in output grid.
 * @param[in]  outRange         Distance from output origin to target (m)
 * @param[in]  inSinSquint      Sine of squint angle in input grid.
 * @param[in]  inCosSquint      Cosine of squint angle in input grid
 *                              Equal to sqrt(1 - inSinSquint^2)
 * @param[in]  inRange          Distance from input origin to target (m)
 * @param[in]  inOrigin         Origin of the input polar grid, ECEF XYZ (m)
 * @param[in]  inAxis           Along-track axis of input polar grid, unit ECEF XYZ
 * @param[in]  outOrigin        Origin of the output polar grid, ECEF XYZ (m)
 * @param[in]  outAxis          Along-track axis of output polar grid, unit ECEF XYZ
 * @param[in]  dem              Digital elevation model (m above ellipsoid)
 * @param[in]  ellipsoid        Ellipsoid associated with DEM
 * @param[in]  side             Look direction (Left or Right)
 * @param[in]  params           Root finding algorithm parameters
 */
template<class DEMInterpolator>
CUDA_HOSTDEV isce3::error::ErrorCode
polar2polar_bracket(double* outSinSquint, double* outRange,
        const double inSinSquint, const double inCosSquint,
        const double inRange,
        const isce3::core::Vec3& inOrigin, const isce3::core::Vec3& inAxis,
        const isce3::core::Vec3& outOrigin, const isce3::core::Vec3& outAxis,
        const DEMInterpolator& dem, const isce3::core::Ellipsoid& ellipsoid,
        const isce3::core::LookSide& side, const Rdr2GeoBracketParams& params);

}}} // namespace isce3::geometry::detail

#include "Rdr2Geo.icc"
