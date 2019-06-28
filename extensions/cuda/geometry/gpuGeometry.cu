//
// Author: Bryan Riel
// Copyright 2018
//

#include "gpuGeometry.h"
#include <isce/cuda/except/Error.h>
#include <isce/core/Basis.h>

using isce::core::Basis;
using isce::core::Vec3;

/** @param[in] pixel Pixel object
 * @param[in] TCNbasis Geocentric TCN basis corresponding to pixel
 * @param[in] pos/vel state vector position/velocity vectors
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] demInterp gpuDEMInterpolator object
 * @param[out] targetLLH output Lon/Lat/Hae corresponding to pixel
 * @param[in] side +1 for left and -1 for right
 * @param[in] threshold Distance threshold for convergence
 * @param[in] maxIter Number of primary iterations
 * @param[in] extraIter Number of secondary iterations
 *
 * This is the elementary device-side transformation from radar geometry to map geometry. The transformation is applicable for a single slant range and azimuth time. The slant range and Doppler information are encapsulated in the Pixel object, so this function can work for both zero and native Doppler geometries. The azimuth time information is encapsulated in the TCNbasis of the platform. For algorithmic details, see \ref overview_geometry "geometry overview".*/
CUDA_DEV
int isce::cuda::geometry::
rdr2geo(const isce::core::Pixel & pixel,
        const Basis& TCNbasis,
        const Vec3& pos, const Vec3& vel,
        const isce::core::Ellipsoid& ellipsoid,
        const gpuDEMInterpolator & demInterp,
        Vec3& targetLLH,
        int side, double threshold, int maxIter, int extraIter) {

    // Initialization
    Vec3 targetLLH_old, targetVec_old, lookVec;

    // Compute normalized velocity
    const Vec3 vhat = vel.unitVec();

    // Unpack TCN basis vectors to pointers
    const auto& that = TCNbasis.x0();
    const auto& chat = TCNbasis.x1();
    const auto& nhat = TCNbasis.x2();

    // Pre-compute TCN vector products
    const double ndotv = nhat.dot(vhat);
    const double vdott = vhat.dot(that);

    // Compute major and minor axes of ellipsoid
    const double major = ellipsoid.a();
    const double minor = major * std::sqrt(1.0 - ellipsoid.e2());

    // Set up orthonormal system right below satellite
    const double satDist = pos.norm();
    const double eta = 1.0 / std::sqrt(
        std::pow(pos[0] / major, 2) +
        std::pow(pos[1] / major, 2) +
        std::pow(pos[2] / minor, 2)
    );
    const double radius = eta * satDist;
    const double hgt = (1.0 - eta) * satDist;

    // Iterate
    int converged = 0;
    double zrdr = targetLLH[2];
    for (int i = 0; i < (maxIter + extraIter); ++i) {

        // Near nadir test
        if ((hgt - zrdr) >= pixel.range())
            break;

        // Cache the previous solution
        for (int k = 0; k < 3; ++k) {
            targetLLH_old[k] = targetLLH[k];
        }

        // Compute angles
        const double a = satDist;
        const double b = radius + zrdr;
        const double costheta = 0.5 * (a / pixel.range() + pixel.range() / a
                              - (b/a) * (b/pixel.range()));
        const double sintheta = std::sqrt(1.0 - costheta*costheta);

        // Compute TCN scale factors
        const double gamma = pixel.range() * costheta;
        const double alpha = (pixel.dopfact() - gamma * ndotv) / vdott;
        const double beta = -side * std::sqrt(std::pow(pixel.range(), 2)
                                            * std::pow(sintheta, 2)
                                            - std::pow(alpha, 2));

        // Compute vector from satellite to ground
        const Vec3 delta = alpha * that + beta * chat + gamma * nhat;
        Vec3 targetVec = pos + delta;

        // Compute LLH of ground point
        ellipsoid.xyzToLonLat(targetVec, targetLLH);

        // Interpolate DEM at current lat/lon point
        targetLLH[2] = demInterp.interpolateLonLat(targetLLH[0], targetLLH[1]);

        // Convert back to XYZ with interpolated height
        ellipsoid.lonLatToXyz(targetLLH, targetVec);
        // Compute updated target height
        zrdr = targetVec.norm() - radius;

        // Check convergence
        lookVec = pos - targetVec;
        const double rdiff = pixel.range() - lookVec.norm();
        if (std::abs(rdiff) < threshold) {
            converged = 1;
            break;
        // May need to perform extra iterations
        } else if (i > maxIter) {
            // XYZ position of old solution
            ellipsoid.lonLatToXyz(targetLLH_old, targetVec_old);
            // XYZ position of updated solution
            for (int idx = 0; idx < 3; ++idx)
                targetVec[idx] = 0.5 * (targetVec_old[idx] + targetVec[idx]);
            // Repopulate lat, lon, z
            ellipsoid.xyzToLonLat(targetVec, targetLLH);
            // Compute updated target height
            zrdr = targetVec.norm() - radius;
        }
    }

    // ----- Final computation: output points exactly at range pixel if converged

    // Compute angles
    const double a = satDist;
    const double b = radius + zrdr;
    const double costheta = 0.5 * (a / pixel.range() + pixel.range() / a
                          - (b/a) * (b/pixel.range()));
    const double sintheta = std::sqrt(1.0 - costheta*costheta);

    // Compute TCN scale factors
    const double gamma = pixel.range() * costheta;
    const double alpha = (pixel.dopfact() - gamma * ndotv) / vdott;
    const double beta = -side * std::sqrt(std::pow(pixel.range(), 2)
                                        * std::pow(sintheta, 2)
                                        - std::pow(alpha, 2));

    // Compute vector from satellite to ground
    const Vec3 delta = alpha * that + beta * chat + gamma * nhat;
    const Vec3 targetVec = pos + delta;

    // Compute LLH of ground point
    targetLLH = ellipsoid.xyzToLonLat(targetVec);

    // Return convergence flag
    return converged;
}

/*
 * This is a helper function for calling the more primitive version of rdr2geo.
 * For more information, see its CPU equivalent in isce/geometry/geometry.h.
 * This GPU version is simplified since it cannot perform error checking.
 */
__device__ int isce::cuda::geometry::rdr2geo(
        double aztime, double slant_range, double doppler,
        const isce::cuda::core::gpuOrbit& orbit,
        const isce::core::Ellipsoid& ellipsoid,
        const isce::cuda::geometry::gpuDEMInterpolator& dem_interp,
        Vec3& target_llh, double wvl, int side, double threshold,
        int max_iter, int extra_iter) {

    /*
     * Interpolate Orbit to azimuth time, compute TCN basis,
     * and estimate geographic coordinates.
     */

    // Interpolate orbit to get state vector
    Vec3 pos, vel;
    orbit.interpolateWGS84Orbit(aztime, pos.data(), vel.data());

    // Set up geocentric TCN basis
    const Basis tcn_basis(pos, vel);

    // Compute satellite velocity magnitude
    const double vmag = vel.norm();

    // Compute Doppler factor
    const double dopfact = 0.5 * wvl * doppler * slant_range / vmag;

    // Wrap range and Doppler factor in a Pixel object
    isce::core::Pixel pixel(slant_range, dopfact, 0);

    // Finally, call rdr2geo
    return rdr2geo(pixel, tcn_basis, pos, vel, ellipsoid, dem_interp,
                   target_llh, side, threshold, max_iter, extra_iter);
}

/** @param[in] inputLLH Lon/Lat/Hae of target of interest
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] orbit gpuOrbit object
 * @param[in] doppler gpuLUT1d Doppler model
 * @param[out] aztime azimuth time of inputLLH w.r.t reference epoch of the orbit
 * @param[out] slantRange slant range to inputLLH
 * @param[in] wavelength Radar wavelength
 * @param[in] threshold azimuth time convergence threshold in seconds
 * @param[in] maxIter Maximum number of Newton-Raphson iterations
 * @param[in] deltaRange step size used for computing derivative of doppler
 *
 * This is the elementary device-side transformation from map geometry to radar geometry. The transformation is applicable for a single lon/lat/h coordinate (i.e., a single point target). For algorithmic details, see \ref overview_geometry "geometry overview".*/
CUDA_DEV
int isce::cuda::geometry::
geo2rdr(const Vec3& inputLLH,
        const isce::core::Ellipsoid& ellipsoid,
        const isce::cuda::core::gpuOrbit& orbit,
        const isce::cuda::core::gpuLUT1d<double> & doppler,
        double * aztime_result, double * slantRange_result,
        double wavelength, double threshold, int maxIter, double deltaRange) {

    // Cartesian type local variables
    // Temp local variables for results
    double aztime, slantRange;

    // Convert LLH to XYZ
    const Vec3 inputXYZ = ellipsoid.lonLatToXyz(inputLLH);

    // Pre-compute scale factor for doppler
    const double dopscale = 0.5 * wavelength;

    // Use mid-orbit epoch as initial guess
    aztime = orbit.UTCtime[orbit.nVectors / 2];

    // Begin iterations
    int converged = 0;
    double slantRange_old = 0.0;
    for (int i = 0; i < maxIter; ++i) {

        // Interpolate the orbit to current estimate of azimuth time
        Vec3 pos, vel;
        orbit.interpolateWGS84Orbit(aztime, &pos[0], &vel[0]);

        // Compute slant range from satellite to ground point
        const Vec3 dr = inputXYZ - pos;
        slantRange = dr.norm();
        // Check convergence
        if (std::abs(slantRange - slantRange_old) < threshold) {
            converged = 1;
            *slantRange_result = slantRange;
            *aztime_result = aztime;
            return converged;
        } else {
            slantRange_old = slantRange;
        }

        // Compute doppler
        const double dopfact = dr.dot(vel);
        const double fdop = doppler.eval(slantRange) * dopscale;
        // Use forward difference to compute doppler derivative
        const double fdopder = (doppler.eval(slantRange + deltaRange) * dopscale - fdop)
                             / deltaRange;

        // Evaluate cost function and its derivative
        const double fn = dopfact - fdop * slantRange;
        const double c1 = -vel.dot(vel);
        const double c2 = (fdop / slantRange) + fdopder;
        const double fnprime = c1 + c2 * dopfact;

        // Update guess for azimuth time
        aztime -= fn / fnprime;
    }

    // If we reach this point, no convergence for specified threshold
    *slantRange_result = slantRange;
    *aztime_result = aztime;
    return converged;

}

// Create ProjectionBase pointer on the device (meant to be run by a single thread)
__global__
void
createProjection(isce::cuda::core::ProjectionBase ** proj, int epsgCode) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*proj) = isce::cuda::core::createProj(epsgCode);
    }
}

// Delete ProjectionBase pointer on the device (meant to be run by a single thread)
__global__
void
deleteProjection(isce::cuda::core::ProjectionBase ** proj) {
    delete *proj;
}

// Helper kernel to call device-side rdr2geo
__global__
void rdr2geo_d(const isce::core::Pixel pixel,
               const Basis TCNbasis,
               const Vec3 pos, const Vec3 vel,
               const isce::core::Ellipsoid ellipsoid,
               isce::cuda::geometry::gpuDEMInterpolator demInterp,
               Vec3* targetLLH,
               int side, double threshold, int maxIter, int extraIter,
               int *resultcode) {

    // Call device function
    *resultcode = isce::cuda::geometry::rdr2geo(
        pixel, TCNbasis, pos, vel, ellipsoid, demInterp, *targetLLH, side,
        threshold, maxIter, extraIter);

}

// Host radar->geo to test underlying functions in a single-threaded context
CUDA_HOST
int isce::cuda::geometry::
rdr2geo_h(const isce::core::Pixel & pixel,
          const Basis& basis,
          const Vec3& pos, const Vec3& vel,
          const isce::core::Ellipsoid & ellipsoid,
          isce::geometry::DEMInterpolator & demInterp,
          Vec3& llh,
          int side, double threshold, int maxIter, int extraIter) {

    // Make GPU objects
    isce::cuda::geometry::gpuDEMInterpolator gpu_demInterp(demInterp);

    // Allocate device memory
    Vec3* llh_d;
    int * resultcode_d;
    cudaMalloc((double **) &llh_d, 3*sizeof(double));
    cudaMalloc((int **) &resultcode_d, sizeof(int));

    // Copy initial values
    cudaMemcpy(llh_d, llh.data(), 3*sizeof(double), cudaMemcpyHostToDevice);

    // DEM interpolator initializes its projection and interpolator
    gpu_demInterp.initProjInterp();

    // Run the rdr2geo on the GPU
    dim3 grid(1), block(1);
    rdr2geo_d<<<grid, block>>>(pixel, basis, pos, vel, ellipsoid,
                               gpu_demInterp, llh_d, side, threshold, maxIter,
                               extraIter, resultcode_d);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());

    // Delete projection pointer on device
    gpu_demInterp.finalizeProjInterp();

    // Copy the resulting llh back to the CPU
    int resultcode;
    checkCudaErrors(cudaMemcpy(llh.data(), llh_d, 3*sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&resultcode, resultcode_d, sizeof(int), cudaMemcpyDeviceToHost));

    // Free memory
    checkCudaErrors(cudaFree(llh_d));
    checkCudaErrors(cudaFree(resultcode_d));

    // Return result code
    return resultcode;
}

// Helper kernel to call device-side geo2rdr
__global__
void geo2rdr_d(const Vec3 llh,
               isce::core::Ellipsoid ellps,
               isce::cuda::core::gpuOrbit orbit,
               isce::cuda::core::gpuLUT1d<double> doppler,
               double * aztime, double * slantRange,
               double wavelength, double threshold, int maxIter, double deltaRange,
               int *resultcode) {

    // Call device function
    *resultcode = isce::cuda::geometry::geo2rdr(
        llh, ellps, orbit, doppler, aztime, slantRange, wavelength, threshold,
        maxIter, deltaRange);
}

// Host geo->radar to test underlying functions in a single-threaded context
CUDA_HOST
int isce::cuda::geometry::
geo2rdr_h(const cartesian_t& llh,
          const isce::core::Ellipsoid & ellps,
          const isce::core::Orbit & orbit,
          const isce::core::LUT1d<double> & doppler,
          double & aztime, double & slantRange,
          double wavelength, double threshold, int maxIter, double deltaRange) {

    // Make GPU objects
    isce::core::Ellipsoid gpu_ellps(ellps);
    isce::cuda::core::gpuOrbit gpu_orbit(orbit);
    isce::cuda::core::gpuLUT1d<double> gpu_doppler(doppler);

    // Allocate necessary device memory
    double *llh_d, *aztime_d, *slantRange_d;
    int *resultcode_d;
    cudaMalloc((double **) &llh_d, 3*sizeof(double));
    cudaMalloc((double **) &aztime_d, sizeof(double));
    cudaMalloc((double **) &slantRange_d, sizeof(double));
    cudaMalloc((int **) &resultcode_d, sizeof(int));

    // Copy input values
    cudaMemcpy(llh_d, llh.data(), 3*sizeof(double), cudaMemcpyHostToDevice);

    // Run geo2rdr on the GPU
    dim3 grid(1), block(1);
    geo2rdr_d<<<grid, block>>>(llh, gpu_ellps, gpu_orbit, gpu_doppler, aztime_d, slantRange_d,
                               wavelength, threshold, maxIter, deltaRange, resultcode_d);

    // Copy results to CPU and return any error code
    int resultcode;
    cudaMemcpy(&aztime, aztime_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&slantRange, slantRange_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&resultcode, resultcode_d, sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(llh_d);
    cudaFree(aztime_d);
    cudaFree(slantRange_d);
    cudaFree(resultcode_d);

    // Return error code
    return resultcode;
}
