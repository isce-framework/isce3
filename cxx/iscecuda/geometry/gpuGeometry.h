#pragma once

#include <isce/core/forward.h>
#include <isce/cuda/core/forward.h>
#include <isce/cuda/geometry/forward.h>
#include <isce/geometry/forward.h>

#include <isce/core/Common.h>

namespace isce { namespace cuda { namespace geometry {

using cartesian_t = isce::core::Vec3;

/**
 * Radar geometry coordinates to map coordinates transformer
 *
 * This is a helper function for calling the more primitive version of rdr2geo.
 * For more information, see its CPU equivalent in isce/geometry/geometry.h.
 * This GPU version is simplified since it cannot perform error checking.
 */
CUDA_DEV int rdr2geo(double, double, double, const isce::cuda::core::OrbitView&,
                     const isce::core::Ellipsoid&, const gpuDEMInterpolator&,
                     isce::core::Vec3&, double, isce::core::LookSide, double,
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
CUDA_DEV int rdr2geo(const isce::core::Pixel& pixel,
                     const isce::core::Basis& TCNbasis,
                     const isce::core::Vec3& pos, const isce::core::Vec3& vel,
                     const isce::core::Ellipsoid& ellipsoid,
                     const gpuDEMInterpolator& demInterp,
                     isce::core::Vec3& targetLLH, isce::core::LookSide side,
                     double threshold, int maxIter, int extraIter);

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
CUDA_DEV int geo2rdr(const isce::core::Vec3& inputLLH,
                     const isce::core::Ellipsoid& ellipsoid,
                     const isce::cuda::core::OrbitView& orbit,
                     const isce::cuda::core::gpuLUT1d<double>& doppler,
                     double* aztime, double* slantRange, double wavelength,
                     isce::core::LookSide side, double threshold, int maxIter,
                     double deltaRange);

/** Radar geometry coordinates to map coordinates transformer (host testing) */
CUDA_HOST int rdr2geo_h(const isce::core::Pixel&, const isce::core::Basis&,
                        const isce::core::Vec3& pos,
                        const isce::core::Vec3& vel,
                        const isce::core::Ellipsoid&,
                        isce::geometry::DEMInterpolator&, cartesian_t&,
                        isce::core::LookSide, double, int, int);

/** Map coordinates to radar geometry coordinates transformer (host testing) */
CUDA_HOST int geo2rdr_h(const cartesian_t&, const isce::core::Ellipsoid&,
                        const isce::core::Orbit&,
                        const isce::core::LUT1d<double>&, double&, double&,
                        double, isce::core::LookSide, double, int, double);

}}} // namespace isce::cuda::geometry

/** Create ProjectionBase pointer on the device (meant to be run by a single
 * thread) */
CUDA_GLOBAL void createProjection(isce::cuda::core::ProjectionBase**, int);

/** Delete ProjectionBase pointer on the device (meant to be run by a single
 * thread) */
CUDA_GLOBAL void deleteProjection(isce::cuda::core::ProjectionBase**);
