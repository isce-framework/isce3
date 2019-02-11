//
// Author: Bryan Riel
// Copyright 2018
//

#include "gpuGeometry.h"
#include "../helper_cuda.h"
#include "isce/cuda/core/gpuPixel.h"

using isce::cuda::core::gpuLinAlg;

/** @param[in] pixel gpuPixel object
 * @param[in] TCNbasis Geocentric TCN basis corresponding to pixel
 * @param[in] state gpuStateVector object
 * @param[in] ellipsoid gpuEllipsoid object
 * @param[in] demInterp gpuDEMInterpolator object
 * @param[out] targetLLH output Lon/Lat/Hae corresponding to pixel
 * @param[in] side +1 for left and -1 for right
 * @param[in] threshold Distance threshold for convergence
 * @param[in] maxIter Number of primary iterations
 * @param[in] extraIter Number of secondary iterations
 *
 * This is the elementary device-side transformation from radar geometry to map geometry. The transformation is applicable for a single slant range and azimuth time. The slant range and Doppler information are encapsulated in the Pixel object, so this function can work for both zero and native Doppler geometries. The azimuth time information is encapsulated in the TCNbasis and StateVector of the platform. For algorithmic details, see \ref overview_geometry "geometry overview".*/
CUDA_DEV
int isce::cuda::geometry::
rdr2geo(const isce::cuda::core::gpuPixel & pixel,
        const isce::cuda::core::gpuBasis & TCNbasis,
        const isce::cuda::core::gpuStateVector & state,
        const isce::cuda::core::gpuEllipsoid & ellipsoid,
        const gpuDEMInterpolator & demInterp,
        double * targetLLH,
        int side, double threshold, int maxIter, int extraIter) {

    // Initialization
    double targetVec[3], targetLLH_old[3], targetVec_old[3],
           lookVec[3], delta[3], delta_temp[3], vhat[3];

    // Compute normalized velocity
    gpuLinAlg::unitVec(state.velocity, vhat);

    // Unpack TCN basis vectors to pointers
    const double * that = TCNbasis.x0;
    const double * chat = TCNbasis.x1;
    const double * nhat = TCNbasis.x2;

    // Pre-compute TCN vector products
    const double ndotv = nhat[0]*vhat[0] + nhat[1]*vhat[1] + nhat[2]*vhat[2];
    const double vdott = vhat[0]*that[0] + vhat[1]*that[1] + vhat[2]*that[2];

    // Compute major and minor axes of ellipsoid
    const double major = ellipsoid.a;
    const double minor = major * std::sqrt(1.0 - ellipsoid.e2);

    // Set up orthonormal system right below satellite
    const double satDist = gpuLinAlg::norm(state.position);
    const double eta = 1.0 / std::sqrt(
        std::pow(state.position[0] / major, 2) +
        std::pow(state.position[1] / major, 2) +
        std::pow(state.position[2] / minor, 2)
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
        gpuLinAlg::linComb(alpha, that, beta, chat, delta_temp);
        gpuLinAlg::linComb(1.0, delta_temp, gamma, nhat, delta);
        gpuLinAlg::linComb(1.0, state.position, 1.0, delta, targetVec);

        // Compute LLH of ground point
        ellipsoid.xyzToLonLat(targetVec, targetLLH);

        // Interpolate DEM at current lat/lon point
        targetLLH[2] = demInterp.interpolateLonLat(targetLLH[0], targetLLH[1]);

        // Convert back to XYZ with interpolated height
        ellipsoid.lonLatToXyz(targetLLH, targetVec);
        // Compute updated target height
        zrdr = gpuLinAlg::norm(targetVec) - radius;

        // Check convergence
        gpuLinAlg::linComb(1.0, state.position, -1.0, targetVec, lookVec);
        const double rdiff = pixel.range() - gpuLinAlg::norm(lookVec);
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
            zrdr = gpuLinAlg::norm(targetVec) - radius;
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
    gpuLinAlg::linComb(alpha, that, beta, chat, delta_temp);
    gpuLinAlg::linComb(1.0, delta_temp, gamma, nhat, delta);
    gpuLinAlg::linComb(1.0, state.position, 1.0, delta, targetVec);

    // Compute LLH of ground point
    ellipsoid.xyzToLonLat(targetVec, targetLLH);

    // Return convergence flag
    return converged;
}

// Utility function to compute geocentric tcn basis from state vector
__device__ void geocentricTCN(const double* pos, const double* vel,
                              isce::cuda::core::gpuBasis& basis) {
    double t_hat[3], c_hat[3], n_hat[3], temp[3];
    using isce::cuda::core::gpuLinAlg;
    // Compute basis vectors
    gpuLinAlg::unitVec(pos, n_hat);
    gpuLinAlg::scale(n_hat, -1.);
    gpuLinAlg::cross(n_hat, vel, temp);
    gpuLinAlg::unitVec(temp, c_hat);
    gpuLinAlg::cross(c_hat, n_hat, temp);
    gpuLinAlg::unitVec(temp, t_hat);
    // Store in basis object
    for (int i = 0; i < 3; i++) {
        basis.x0[i] = t_hat[i];
        basis.x1[i] = c_hat[i];
        basis.x2[i] = n_hat[i];
    }
}

/*
 * This is a helper function for calling the more primitive version of rdr2geo.
 * For more information, see its CPU equivalent in isce/geometry/geometry.h.
 * This GPU version is simplified since it cannot perform error checking.
 */
__device__ int isce::cuda::geometry::rdr2geo(
        double aztime, double slant_range, double doppler,
        const isce::cuda::core::gpuOrbit& orbit,
        const isce::cuda::core::gpuEllipsoid& ellipsoid,
        const isce::cuda::geometry::gpuDEMInterpolator& dem_interp,
        double* target_llh, double wvl, int side, double threshold,
        int max_iter, int extra_iter) {

    /*
     * Interpolate Orbit to azimuth time, compute TCN basis,
     * and estimate geographic coordinates.
     */

    // Interpolate orbit to get state vector
    double pos[3], vel[3];
    orbit.interpolateWGS84Orbit(aztime, pos, vel);

    // Set up geocentric TCN basis
    isce::cuda::core::gpuBasis tcn_basis;
    geocentricTCN(pos, vel, tcn_basis);

    // Compute satellite velocity magnitude
    const double vmag = isce::cuda::core::gpuLinAlg::norm(vel);

    // Compute Doppler factor
    const double dopfact = 0.5 * wvl * doppler * slant_range / vmag;

    // Wrap range and Doppler factor in a Pixel object
    isce::cuda::core::gpuPixel pixel(slant_range, dopfact, 0);

    isce::cuda::core::gpuStateVector state;
    for (int i = 0; i < 3; i++) {
        state.position[i] = pos[i];
        state.velocity[i] = vel[i];
    }

    // Finally, call rdr2geo
    return rdr2geo(pixel, tcn_basis, state, ellipsoid, dem_interp,
                   target_llh, side, threshold, max_iter, extra_iter);
}

/** @param[in] inputLLH Lon/Lat/Hae of target of interest
 * @param[in] ellipsoid gpuEllipsoid object
 * @param[in] orbit gpuOrbit object
 * @param[in] doppler gpuLUT1d Doppler model
 * @param[in] mode  gpuImageMode object
 * @param[out] aztime azimuth time of inputLLH w.r.t reference epoch of the orbit
 * @param[out] slantRange slant range to inputLLH
 * @param[in] threshold azimuth time convergence threshold in seconds
 * @param[in] maxIter Maximum number of Newton-Raphson iterations
 * @param[in] deltaRange step size used for computing derivative of doppler
 *
 * This is the elementary device-side transformation from map geometry to radar geometry. The transformation is applicable for a single lon/lat/h coordinate (i.e., a single point target). For algorithmic details, see \ref overview_geometry "geometry overview".*/
CUDA_DEV
int isce::cuda::geometry::
geo2rdr(double * inputLLH,
        const isce::cuda::core::gpuEllipsoid & ellipsoid,
        const isce::cuda::core::gpuOrbit & orbit,
        const isce::cuda::core::gpuLUT1d<double> & doppler,
        const isce::cuda::product::gpuImageMode & mode,
        double * aztime_result, double * slantRange_result,
        double threshold, int maxIter, double deltaRange) {

    // Cartesian type local variables
    double inputXYZ[3], satpos[3], satvel[3], dr[3];
    // Temp local variables for results
    double aztime, slantRange;

    // Convert LLH to XYZ
    ellipsoid.lonLatToXyz(inputLLH, inputXYZ);

    // Pre-compute scale factor for doppler
    const double dopscale = 0.5 * mode.wavelength();

    // Use mid-orbit epoch as initial guess
    aztime = orbit.UTCtime[orbit.nVectors / 2];
    
    // Begin iterations
    int converged = 0;
    double slantRange_old = 0.0;
    for (int i = 0; i < maxIter; ++i) {

        // Interpolate the orbit to current estimate of azimuth time
        orbit.interpolateWGS84Orbit(aztime, satpos, satvel);

        // Compute slant range from satellite to ground point
        gpuLinAlg::linComb(1.0, inputXYZ, -1.0, satpos, dr);
        slantRange = gpuLinAlg::norm(dr);
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
        const double dopfact = gpuLinAlg::dot(dr, satvel);
        const double fdop = doppler.eval(slantRange) * dopscale;
        // Use forward difference to compute doppler derivative
        const double fdopder = (doppler.eval(slantRange + deltaRange) * dopscale - fdop)
                             / deltaRange;

        // Evaluate cost function and its derivative
        const double fn = dopfact - fdop * slantRange;
        const double c1 = -1.0 * gpuLinAlg::dot(satvel, satvel);
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
void rdr2geo_d(const isce::cuda::core::gpuPixel pixel,
               const isce::cuda::core::gpuBasis TCNbasis,
               const isce::cuda::core::gpuStateVector state,
               const isce::cuda::core::gpuEllipsoid ellipsoid,
               isce::cuda::geometry::gpuDEMInterpolator demInterp,
               double * targetLLH,
               int side, double threshold, int maxIter, int extraIter,
               int *resultcode) {

    // Call device function
    *resultcode = isce::cuda::geometry::rdr2geo(
        pixel, TCNbasis, state, ellipsoid, demInterp, targetLLH, side,
        threshold, maxIter, extraIter
    );

}

// Host radar->geo to test underlying functions in a single-threaded context
CUDA_HOST
int isce::cuda::geometry::
rdr2geo_h(const isce::core::Pixel & pixel,
          const isce::core::Basis & basis,
          const isce::core::StateVector & state,
          const isce::core::Ellipsoid & ellipsoid,
          isce::geometry::DEMInterpolator & demInterp,
          cartesian_t & llh,
          int side, double threshold, int maxIter, int extraIter) {

    // Make GPU objects
    isce::cuda::core::gpuPixel gpu_pixel(pixel);
    isce::cuda::core::gpuBasis gpu_basis(basis);
    isce::cuda::core::gpuStateVector gpu_state(state);
    isce::cuda::core::gpuEllipsoid gpu_ellps(ellipsoid);
    isce::cuda::geometry::gpuDEMInterpolator gpu_demInterp(demInterp);
        
    // Allocate device memory
    double * llh_d;
    int * resultcode_d;
    cudaMalloc((double **) &llh_d, 3*sizeof(double));
    cudaMalloc((int **) &resultcode_d, sizeof(int));

    // Copy initial values
    cudaMemcpy(llh_d, llh.data(), 3*sizeof(double), cudaMemcpyHostToDevice);

    // DEM interpolator initializes its projection and interpolator
    gpu_demInterp.initProjInterp();
    
    // Run the rdr2geo on the GPU
    dim3 grid(1), block(1);
    rdr2geo_d<<<grid, block>>>(gpu_pixel, gpu_basis, gpu_state, gpu_ellps,
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
void geo2rdr_d(double * llh,
               isce::cuda::core::gpuEllipsoid ellps,
               isce::cuda::core::gpuOrbit orbit,
               isce::cuda::core::gpuLUT1d<double> doppler,
               isce::cuda::product::gpuImageMode mode,
               double * aztime, double * slantRange,
               double threshold, int maxIter, double deltaRange,
               int *resultcode) {

    // Call device function
    *resultcode = isce::cuda::geometry::geo2rdr(
        llh, ellps, orbit, doppler, mode, aztime, slantRange, threshold,
        maxIter, deltaRange
    );
                          
}

// Host geo->radar to test underlying functions in a single-threaded context
CUDA_HOST
int isce::cuda::geometry::
geo2rdr_h(const cartesian_t & llh,
          const isce::core::Ellipsoid & ellps,
          const isce::core::Orbit & orbit,
          const isce::core::LUT1d<double> & doppler,
          const isce::product::ImageMode & mode,
          double & aztime, double & slantRange,
          double threshold, int maxIter, double deltaRange) {

    // Make GPU objects
    isce::cuda::core::gpuEllipsoid gpu_ellps(ellps);
    isce::cuda::core::gpuOrbit gpu_orbit(orbit);
    isce::cuda::core::gpuLUT1d<double> gpu_doppler(doppler);
    isce::cuda::product::gpuImageMode gpu_mode(mode);

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
    geo2rdr_d<<<grid, block>>>(llh_d, gpu_ellps, gpu_orbit, gpu_doppler, gpu_mode,
                               aztime_d, slantRange_d, threshold, maxIter, deltaRange,
                               resultcode_d);

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
