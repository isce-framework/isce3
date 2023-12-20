#pragma once

#include <optional>

#include <isce3/core/forward.h>
#include <isce3/cuda/core/forward.h>
#include <isce3/cuda/geometry/forward.h>
#include <isce3/geometry/forward.h>

#include <isce3/core/Common.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

namespace isce3 { namespace cuda { namespace geometry {

using cartesian_t = isce3::core::Vec3;

/**
 * Radar geometry coordinates to map coordinates transformer
 *
 * This is a helper function for calling the more primitive version of rdr2geo.
 * For more information, see its CPU equivalent in isce/geometry/geometry.h.
 * This GPU version is simplified since it cannot perform error checking.
 */
CUDA_DEV int rdr2geo(double, double, double, const isce3::cuda::core::OrbitView&,
                     const isce3::core::Ellipsoid&, const gpuDEMInterpolator&,
                     isce3::core::Vec3&, double, isce3::core::LookSide, double,
                     int, int);

/**
 * Radar geometry coordinates to map coordinates transformer
 *
 * @param[in] pixel Pixel object
 * @param[in] TCNbasis Geocentric TCN basis corresponding to pixel
 * @param[in] pos/vel state vector position/velocity vectors
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] demInterp gpuDEMInterpolator object
 * @param[out] targetLLH output Lon/Lat/Hae corresponding to pixel
 * @param[in] side LookSide::Left or LookSide::Right
 * @param[in] threshold Distance threshold for convergence
 * @param[in] maxIter Number of primary iterations
 * @param[in] extraIter Number of secondary iterations
 *
 * This is the elementary device-side transformation from radar geometry to map
 * geometry. The transformation is applicable for a single slant range and
 * azimuth time. The slant range and Doppler information are encapsulated in the
 * Pixel object, so this function can work for both zero and native Doppler
 * geometries. The azimuth time information is encapsulated in the TCNbasis of
 * the platform. For algorithmic details, see \ref overview_geometry "geometry
 * overview".
 */
CUDA_DEV int rdr2geo(const isce3::core::Pixel& pixel,
                     const isce3::core::Basis& TCNbasis,
                     const isce3::core::Vec3& pos, const isce3::core::Vec3& vel,
                     const isce3::core::Ellipsoid& ellipsoid,
                     const gpuDEMInterpolator& demInterp,
                     isce3::core::Vec3& targetLLH, isce3::core::LookSide side,
                     double threshold, int maxIter, int extraIter);

/**
 * Radar geometry coordinates to map coordinates transformer using a bracketing
 * algorithm.
 *
 * @param[in] aztime        Azimuth time of target since orbit epoch (s)
 * @param[in] slantRange    Range to target (m)
 * @param[in] doppler       Doppler of target (Hz)
 * @param[in] orbit         Radar trajectory
 * @param[in] ellipsoid     Surface ellipsoid corresponding to DEM
 * @param[in] dem           Digital elevation model
 * @param[out] targetXYZ    Target ECEF XYZ position (m)
 * @param[in] wvl           Wavelength corresponding to Doppler measurement (m)
 * @param[in] side          Side that radar is viewing
 * @param[in] tolHeight     Height convergence tolerance (m)
 * @param[in] lookMin       Smallest possible pseudo-look angle (rad)
 * @param[in] lookMax       Largest possible pseudo-look angle (rad)
 *
 * @returns nonzero on success, zero otherwise
 */
CUDA_DEV int rdr2geo_bracket(double aztime, double slantRange, double doppler,
        const isce3::cuda::core::OrbitView& orbit,
        const isce3::core::Ellipsoid& ellipsoid, const gpuDEMInterpolator& dem,
        isce3::core::Vec3& targetXYZ, double wvl, isce3::core::LookSide side,
        double tolHeight = isce3::geometry::detail::DEFAULT_TOL_HEIGHT,
        double lookMin = 0.0, double lookMax = M_PI / 2);

/**
 * Map coordinates to radar geometry coordinates transformer
 *
 * @param[in] inputLLH Lon/Lat/Hae of target of interest
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] orbit gpuOrbit object
 * @param[in] doppler gpuLUT1d Doppler model
 * @param[out] aztime azimuth time of inputLLH w.r.t reference epoch of the
 * orbit
 * @param[out] slantRange slant range to inputLLH
 * @param[in] wavelength Radar wavelength
 * @param[in] threshold azimuth time convergence threshold in seconds
 * @param[in] maxIter Maximum number of Newton-Raphson iterations
 * @param[in] deltaRange step size used for computing derivative of doppler
 *
 * This is the elementary device-side transformation from map geometry to radar
 * geometry. The transformation is applicable for a single lon/lat/h coordinate
 * (i.e., a single point target). For algorithmic details, see \ref
 * overview_geometry "geometry overview".
 */
CUDA_DEV int geo2rdr(const isce3::core::Vec3& inputLLH,
                     const isce3::core::Ellipsoid& ellipsoid,
                     const isce3::cuda::core::OrbitView& orbit,
                     const isce3::cuda::core::gpuLUT1d<double>& doppler,
                     double* aztime, double* slantRange, double wavelength,
                     isce3::core::LookSide side, double threshold, int maxIter,
                     double deltaRange);

/**
 * Map coordinates to radar geometry coordinates transformer
 *
 * @param[in] inputLLH Lon/Lat/Hae of target of interest
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] orbit gpuOrbit object
 * @param[in] doppler gpuLUT2d Doppler model
 * @param[out] aztime azimuth time of inputLLH w.r.t reference epoch of the
 * orbit
 * @param[out] slantRange slant range to inputLLH
 * @param[in] wavelength Radar wavelength
 * @param[in] threshold slant range convergence threshold in meters
 * @param[in] maxIter Maximum number of Newton-Raphson iterations
 * @param[in] deltaRange step size used for computing derivative of doppler
 *
 * This is the elementary device-side transformation from map geometry to radar
 * geometry. The transformation is applicable for a single lon/lat/h coordinate
 * (i.e., a single point target). For algorithmic details, see \ref
 * overview_geometry "geometry overview".
 */
CUDA_DEV int geo2rdr(const isce3::core::Vec3& inputLLH,
                     const isce3::core::Ellipsoid& ellipsoid,
                     const isce3::cuda::core::OrbitView& orbit,
                     const isce3::cuda::core::gpuLUT2d<double>& doppler,
                     double* aztime, double* slantRange, double wavelength,
                     isce3::core::LookSide side, double threshold, int maxIter,
                     double deltaRange);

/**
 * Map coordinates to radar geometry coordinates transformer using a bracketing
 * algorithm.
 *
 * \param[in]  x          Target ECEF XYZ position (m)
 * \param[in]  orbit      Platform orbit.  Time bounds used as search interval.
 * \param[in]  doppler    Doppler model as a function of azimuth & range (Hz)
 * \param[out] aztime     Target azimuth time w.r.t. orbit reference epoch (s)
 * \param[out] range      Target slant range (m)
 * \param[in]  wavelength Radar wavelength (m)
 * \param[in]  side       Radar look side
 * \param[in]  dt         Allowable error in azimuth time solution (s)
 * \param[in]  timeStart  Start of search interval, s
 *                        Defaults to max of orbit and Doppler LUT start time
 * \param[in]  timeEnd    End of search interval, s
 *                        Defaults to min of orbit and Doppler LUT end time
 */
CUDA_DEV int geo2rdr_bracket(const isce3::core::Vec3& x,
        const isce3::cuda::core::OrbitView& orbit,
        const isce3::cuda::core::gpuLUT2d<double>& doppler, double* aztime,
        double* range, const double wavelength,
        const isce3::core::LookSide side, const double dt,
        std::optional<double> timeStart = std::nullopt,
        std::optional<double> timeEnd = std::nullopt);


/** Radar geometry coordinates to map coordinates transformer (host testing) */
CUDA_HOST int rdr2geo_h(const isce3::core::Pixel&, const isce3::core::Basis&,
                        const isce3::core::Vec3& pos,
                        const isce3::core::Vec3& vel,
                        const isce3::core::Ellipsoid&,
                        isce3::geometry::DEMInterpolator&, cartesian_t&,
                        isce3::core::LookSide, double, int, int);

/** Map coordinates to radar geometry coordinates transformer (host testing) */
CUDA_HOST int geo2rdr_h(const cartesian_t&, const isce3::core::Ellipsoid&,
                        const isce3::core::Orbit&,
                        const isce3::core::LUT1d<double>&, double&, double&,
                        double, isce3::core::LookSide, double, int, double);

CUDA_HOST
int rdr2geo_bracket_h(double aztime, double slantRange, double doppler,
        const isce3::core::Orbit& orbit,
        const isce3::core::Ellipsoid& ellipsoid,
        const isce3::geometry::DEMInterpolator& dem,
        isce3::core::Vec3& targetXYZ, double wavelength,
        isce3::core::LookSide side, double tolHeight, double lookMin,
        double lookMax);

CUDA_HOST int geo2rdr_bracket_h(const isce3::core::Vec3&,
        const isce3::core::Orbit&, const isce3::core::LUT2d<double>&, double&,
        double&, double, isce3::core::LookSide, double,
        std::optional<double> = std::nullopt,
        std::optional<double> = std::nullopt);

}}} // namespace isce3::cuda::geometry

/** Create ProjectionBase pointer on the device (meant to be run by a single
 * thread) */
CUDA_GLOBAL void createProjection(isce3::cuda::core::ProjectionBase**, int);

/** Delete ProjectionBase pointer on the device (meant to be run by a single
 * thread) */
CUDA_GLOBAL void deleteProjection(isce3::cuda::core::ProjectionBase**);
