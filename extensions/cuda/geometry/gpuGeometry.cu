//
// Author: Bryan Riel
// Copyright 2018
//

// isce::cuda::core

#include "gpuGeometry.h"

using isce::cuda::core::gpuLinAlg;

CUDA_DEV
int isce::cuda::geometry::
geo2rdr(double * inputLLH,
        const isce::cuda::core::gpuEllipsoid & ellipsoid,
        const isce::cuda::core::gpuOrbit & orbit,
        const isce::cuda::core::gpuPoly2d & doppler,
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

    // Compute minimum and maximum valid range
    const double rangeMin = mode.startingRange();
    const double rangeMax = rangeMin + mode.rangePixelSpacing() * (mode.width() - 1);

    // Compute azimuth time spacing for coarse grid search 
    const int NUM_AZTIME_TEST = 15;
    const double tstart = orbit.UTCtime[0];
    const double tend = orbit.UTCtime[orbit.nVectors - 1];
    const double delta_t = (tend - tstart) / (1.0 * (NUM_AZTIME_TEST - 1));

    // Find azimuth time with minimum valid range distance to target 
    double slantRange_closest = 1.0e16;
    double aztime_closest = -1000.0;
    for (int k = 0; k < NUM_AZTIME_TEST; ++k) {
        // Interpolate orbit
        aztime = tstart + k * delta_t;
        int status = orbit.interpolateWGS84Orbit(aztime, satpos, satvel);
        if (status != 0)
            continue;
        // Compute slant range
        gpuLinAlg::linComb(1.0, inputXYZ, -1.0, satpos, dr);
        slantRange = gpuLinAlg::norm(dr);
        // Check validity
        if (slantRange < rangeMin)
            continue;
        if (slantRange > rangeMax)
            continue;
        // Update best guess
        if (slantRange < slantRange_closest) {
            slantRange_closest = slantRange;
            aztime_closest = aztime;
        }
    }

    // If we did not find a good guess, use tmid as intial guess
    if (aztime_closest < 0.0) {
        aztime = orbit.UTCtime[orbit.nVectors / 2];
    } else {
        aztime = aztime_closest;
    }

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

        // Compute slant range bin
        const double rbin = (slantRange - mode.startingRange()) / mode.rangePixelSpacing();
        // Compute doppler
        const double dopfact = gpuLinAlg::dot(dr, satvel);
        const double fdop = doppler.eval(0, rbin) * dopscale;
        // Use forward difference to compute doppler derivative
        const double fdopder = (doppler.eval(0, rbin + deltaRange) * dopscale - fdop)
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

// Helper function to call device-side geo2rdr
__global__
void geo2rdr_d(double * llh,
               isce::cuda::core::gpuEllipsoid ellps,
               isce::cuda::core::gpuOrbit orbit,
               isce::cuda::core::gpuPoly2d doppler,
               isce::cuda::product::gpuImageMode mode,
               double * aztime, double * slantRange,
               double threshold, int maxIter, double deltaRange,
               int *resultcode) {

    // Call device function
    *resultcode  = isce::cuda::geometry::geo2rdr(
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
          const isce::core::Poly2d & doppler,
          const isce::product::ImageMode & mode,
          double & aztime, double & slantRange,
          double threshold, int maxIter, double deltaRange) {

    // Set the CUDA device
    cudaSetDevice(0);

    // Make GPU objects
    isce::cuda::core::gpuEllipsoid gpu_ellps(ellps);
    isce::cuda::core::gpuOrbit gpu_orbit(orbit);
    isce::cuda::core::gpuPoly2d gpu_doppler(doppler);
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
