#include "gpuGeometry.h"

#include <isce3/core/Basis.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Pixel.h>
#include <isce3/cuda/core/Orbit.h>
#include <isce3/cuda/core/OrbitView.h>
#include <isce3/cuda/core/gpuLUT1d.h>
#include <isce3/cuda/core/gpuLUT2d.h>
#include <isce3/cuda/except/Error.h>
#include <isce3/cuda/geometry/gpuDEMInterpolator.h>
#include <isce3/geometry/detail/Geo2Rdr.h>
#include <isce3/geometry/detail/Rdr2Geo.h>

namespace detail = isce3::geometry::detail;

using isce3::core::Basis;
using isce3::core::LookSide;
using isce3::core::OrbitInterpBorderMode;
using isce3::core::Vec3;
using isce3::error::ErrorCode;

namespace isce3 { namespace cuda { namespace geometry {

CUDA_DEV
int rdr2geo(const isce3::core::Pixel& pixel, const Basis& TCNbasis,
            const Vec3& pos, const Vec3& vel,
            const isce3::core::Ellipsoid& ellipsoid,
            const gpuDEMInterpolator& demInterp, Vec3& targetLLH, LookSide side,
            double threshold, int maxIter, int extraIter)
{
    double h0 = targetLLH[2];
    detail::Rdr2GeoParams params = {threshold, maxIter, extraIter};
    auto status = detail::rdr2geo(&targetLLH, pixel, TCNbasis, pos, vel,
                                  demInterp, ellipsoid, side, h0, params);
    return (status == ErrorCode::Success);
}

__device__ int rdr2geo(double aztime, double slant_range, double doppler,
                       const isce3::cuda::core::OrbitView& orbit,
                       const isce3::core::Ellipsoid& ellipsoid,
                       const gpuDEMInterpolator& dem_interp, Vec3& target_llh,
                       double wvl, LookSide side, double threshold,
                       int max_iter, int extra_iter)
{
    double h0 = target_llh[2];
    detail::Rdr2GeoParams params = {threshold, max_iter, extra_iter};
    auto status =
            detail::rdr2geo(&target_llh, aztime, slant_range, doppler, orbit,
                            dem_interp, ellipsoid, wvl, side, h0, params);
    return (status == ErrorCode::Success);
}

CUDA_DEV int rdr2geo_bracket(double aztime, double slantRange, double doppler,
                       const isce3::cuda::core::OrbitView& orbit,
                       const isce3::core::Ellipsoid& ellipsoid,
                       const gpuDEMInterpolator& dem, Vec3& targetXYZ,
                       double wvl, LookSide side, double tolHeight,
                       double lookMin, double lookMax)
{
    auto status = detail::rdr2geo_bracket(&targetXYZ, aztime, slantRange,
            doppler, orbit, dem, ellipsoid, wvl, side,
            {tolHeight, lookMin, lookMax});
    return status == ErrorCode::Success;
}

CUDA_DEV
int geo2rdr(const Vec3& inputLLH, const isce3::core::Ellipsoid& ellipsoid,
            const isce3::cuda::core::OrbitView& orbit,
            const isce3::cuda::core::gpuLUT1d<double>& doppler,
            double* aztime_result, double* slantRange_result, double wavelength,
            LookSide side, double threshold, int maxIter, double deltaRange)
{

    // Cartesian type local variables
    // Temp local variables for results
    double aztime, slantRange;

    // Convert LLH to XYZ
    const Vec3 inputXYZ = ellipsoid.lonLatToXyz(inputLLH);

    // Pre-compute scale factor for doppler
    const double dopscale = 0.5 * wavelength;

    // Use mid-orbit epoch as initial guess
    aztime = orbit.midTime();

    // Newton step, init to zero.
    double dt = 0.0;

    // Begin iterations
    int converged = 0;
    for (int i = 0; i < maxIter; ++i) {
        // Apply Newton step from previous iteration.
        aztime -= dt;

        // Interpolate the orbit to current estimate of azimuth time
        Vec3 pos, vel;
        orbit.interpolate(&pos, &vel, aztime, OrbitInterpBorderMode::FillNaN);

        // Compute slant range from satellite to ground point
        const Vec3 dr = inputXYZ - pos;
        slantRange = dr.norm();

        // Check look side
        // (Left && positive) || (Right && negative)
        if ((side == LookSide::Right) ^ (dr.cross(vel).dot(pos) > 0)) {
            *slantRange_result = slantRange;
            *aztime_result = aztime;
            return converged;
        }

        // Compute doppler
        const double dopfact = dr.dot(vel);
        const double fdop = doppler.eval(slantRange) * dopscale;
        // Use forward difference to compute doppler derivative
        const double fdopder =
                (doppler.eval(slantRange + deltaRange) * dopscale - fdop) /
                deltaRange;

        // Evaluate cost function and its derivative
        const double fn = dopfact - fdop * slantRange;
        const double c1 = -vel.dot(vel);
        const double c2 = (fdop / slantRange) + fdopder;
        const double fnprime = c1 + c2 * dopfact;

        // Compute Newton step.  Don't apply until start of next iteration so
        // that returned (slantRange, aztime) are always consistent.
        dt = fn / fnprime;

        // Check convergence
        if (std::abs(dt) < threshold) {
            converged = 1;
            *slantRange_result = slantRange;
            *aztime_result = aztime;
            return converged;
        }
    }

    // If we reach this point, no convergence for specified threshold
    *slantRange_result = slantRange;
    *aztime_result = aztime;
    return converged;
}

CUDA_DEV int geo2rdr(const isce3::core::Vec3& inputLLH,
                     const isce3::core::Ellipsoid& ellipsoid,
                     const isce3::cuda::core::OrbitView& orbit,
                     const isce3::cuda::core::gpuLUT2d<double>& doppler,
                     double* aztime, double* slantRange, double wavelength,
                     isce3::core::LookSide side, double threshold, int maxIter,
                     double deltaRange)
{
    double t0 = *aztime;
    detail::Geo2RdrParams params = {threshold, maxIter, deltaRange};
    auto status =
            detail::geo2rdr(aztime, slantRange, inputLLH, ellipsoid, orbit,
                            doppler, wavelength, side, t0, params);
    return (status == ErrorCode::Success);
}


CUDA_DEV int geo2rdr_bracket(
                    const Vec3& x,
                    const isce3::cuda::core::OrbitView& orbit,
                    const isce3::cuda::core::gpuLUT2d<double>& doppler,
                    double* aztime,
                    double* range, const double wavelength,
                    const isce3::core::LookSide side,
                    const double dt,
                    std::optional<double> timeStart,
                    std::optional<double> timeEnd)
{
    ErrorCode err = detail::geo2rdr_bracket(
            aztime, range, x, orbit, doppler, wavelength, side,
            {dt, timeStart, timeEnd});
    return err == ErrorCode::Success;
}

}}} // namespace isce3::cuda::geometry

// Create ProjectionBase pointer on the device (meant to be run by a single
// thread)
__global__ void createProjection(isce3::cuda::core::ProjectionBase** proj,
                                 int epsgCode)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*proj) = isce3::cuda::core::createProj(epsgCode);
    }
}

// Delete ProjectionBase pointer on the device (meant to be run by a single
// thread)
__global__ void deleteProjection(isce3::cuda::core::ProjectionBase** proj)
{
    delete *proj;
}

namespace isce3 { namespace cuda { namespace geometry {

// Helper kernel to call device-side rdr2geo
__global__ void rdr2geo_d(const isce3::core::Pixel pixel, const Basis TCNbasis,
                          const Vec3 pos, const Vec3 vel,
                          const isce3::core::Ellipsoid ellipsoid,
                          gpuDEMInterpolator demInterp, Vec3* targetLLH,
                          LookSide side, double threshold, int maxIter,
                          int extraIter, int* resultcode)
{

    // Call device function
    *resultcode = rdr2geo(pixel, TCNbasis, pos, vel, ellipsoid, demInterp,
                          *targetLLH, side, threshold, maxIter, extraIter);
}

// Host radar->geo to test underlying functions in a single-threaded context
CUDA_HOST
int rdr2geo_h(const isce3::core::Pixel& pixel, const Basis& basis,
              const Vec3& pos, const Vec3& vel,
              const isce3::core::Ellipsoid& ellipsoid,
              isce3::geometry::DEMInterpolator& demInterp, Vec3& llh,
              LookSide side, double threshold, int maxIter, int extraIter)
{

    // Make GPU objects
    gpuDEMInterpolator gpu_demInterp(demInterp);

    // Allocate device memory
    Vec3* llh_d;
    int* resultcode_d;
    cudaMalloc((double**) &llh_d, 3 * sizeof(double));
    cudaMalloc((int**) &resultcode_d, sizeof(int));

    // Copy initial values
    cudaMemcpy(llh_d, llh.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);

    // Run the rdr2geo on the GPU
    dim3 grid(1), block(1);
    rdr2geo_d<<<grid, block>>>(pixel, basis, pos, vel, ellipsoid, gpu_demInterp,
                               llh_d, side, threshold, maxIter, extraIter,
                               resultcode_d);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());

    // Copy the resulting llh back to the CPU
    int resultcode;
    checkCudaErrors(cudaMemcpy(llh.data(), llh_d, 3 * sizeof(double),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&resultcode, resultcode_d, sizeof(int),
                               cudaMemcpyDeviceToHost));

    // Free memory
    checkCudaErrors(cudaFree(llh_d));
    checkCudaErrors(cudaFree(resultcode_d));

    // Return result code
    return resultcode;
}

// Helper kernel to call device-side geo2rdr
__global__ void geo2rdr_d(const Vec3 llh, isce3::core::Ellipsoid ellps,
                          isce3::cuda::core::OrbitView orbit,
                          isce3::cuda::core::gpuLUT1d<double> doppler,
                          double* aztime, double* slantRange, double wavelength,
                          LookSide side, double threshold, int maxIter,
                          double deltaRange, int* resultcode)
{

    // Call device function
    *resultcode = geo2rdr(llh, ellps, orbit, doppler, aztime, slantRange,
                          wavelength, side, threshold, maxIter, deltaRange);
}

// Host geo->radar to test underlying functions in a single-threaded context
CUDA_HOST
int geo2rdr_h(const cartesian_t& llh, const isce3::core::Ellipsoid& ellps,
              const isce3::core::Orbit& orbit,
              const isce3::core::LUT1d<double>& doppler, double& aztime,
              double& slantRange, double wavelength, LookSide side,
              double threshold, int maxIter, double deltaRange)
{

    // Make GPU objects
    isce3::core::Ellipsoid gpu_ellps(ellps);
    isce3::cuda::core::Orbit gpu_orbit(orbit);
    isce3::cuda::core::gpuLUT1d<double> gpu_doppler(doppler);

    // Allocate necessary device memory
    double *llh_d, *aztime_d, *slantRange_d;
    int* resultcode_d;
    cudaMalloc((double**) &llh_d, 3 * sizeof(double));
    cudaMalloc((double**) &aztime_d, sizeof(double));
    cudaMalloc((double**) &slantRange_d, sizeof(double));
    cudaMalloc((int**) &resultcode_d, sizeof(int));

    // Copy input values
    cudaMemcpy(llh_d, llh.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);

    // Run geo2rdr on the GPU
    dim3 grid(1), block(1);
    geo2rdr_d<<<grid, block>>>(llh, gpu_ellps, gpu_orbit, gpu_doppler, aztime_d,
                               slantRange_d, wavelength, side, threshold,
                               maxIter, deltaRange, resultcode_d);

    // Copy results to CPU and return any error code
    int resultcode;
    cudaMemcpy(&aztime, aztime_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&slantRange, slantRange_d, sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&resultcode, resultcode_d, sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(llh_d);
    cudaFree(aztime_d);
    cudaFree(slantRange_d);
    cudaFree(resultcode_d);

    // Return error code
    return resultcode;
}

// Helper kernel to call device-side geo2rdr
__global__ void geo2rdr_bracket_d(const Vec3 x,
                          isce3::cuda::core::OrbitView orbit,
                          isce3::cuda::core::gpuLUT2d<double> doppler,
                          double* aztime, double* slantRange, double wavelength,
                          const LookSide side, double dt,
                          std::optional<double> timeStart,
                          std::optional<double> timeEnd, int* err)
{

    // Call device function
    *err = geo2rdr_bracket(x, orbit, doppler, aztime, slantRange,
                          wavelength, side, dt, timeStart, timeEnd);
}

// Host geo->radar to test underlying functions in a single-threaded context
CUDA_HOST
int geo2rdr_bracket_h(const cartesian_t& x,
              const isce3::core::Orbit& orbit,
              const isce3::core::LUT2d<double>& doppler, double& aztime,
              double& slantRange, double wavelength, LookSide side,
              double threshold,
              std::optional<double> timeStart, std::optional<double> timeEnd)
{
    // Make GPU objects
    isce3::cuda::core::Orbit gpu_orbit(orbit);
    isce3::cuda::core::gpuLUT2d<double> gpu_doppler(doppler);

    // Allocate necessary device memory
    double *aztime_d, *slantRange_d;
    int* err_d;
    checkCudaErrors(cudaMalloc((double**) &aztime_d, sizeof(double)));
    checkCudaErrors(cudaMalloc((double**) &slantRange_d, sizeof(double)));
    checkCudaErrors(cudaMalloc((int**) &err_d, sizeof(int)));

    // Run geo2rdr on the GPU
    dim3 grid(1), block(1);
    geo2rdr_bracket_d<<<grid, block>>>(x, gpu_orbit, gpu_doppler, aztime_d,
                               slantRange_d, wavelength, side, threshold,
                               timeStart, timeEnd, err_d);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaStreamSynchronize(0));

    // Copy results to CPU and return any error code
    int err;
    checkCudaErrors(cudaMemcpy(&aztime, aztime_d, sizeof(double), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&slantRange, slantRange_d, sizeof(double),
               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&err, err_d, sizeof(int), cudaMemcpyDeviceToHost));

    // Free memory
    cudaFree(aztime_d);
    cudaFree(slantRange_d);
    cudaFree(err_d);

    // Return error code
    return err;
}

// Helper kernel to call device-side rdr2geo
__global__ void rdr2geo_bracket_d(double aztime, double slantRange,
                        double doppler,
                        isce3::cuda::core::OrbitView orbit,
                        isce3::core::Ellipsoid ellipsoid,
                        isce3::cuda::geometry::gpuDEMInterpolator dem,
                        Vec3* targetXYZ, double wavelength,
                        const LookSide side, double tolHeight, double lookMin,
                        double lookMax, int* err)
{

    // Call device function
    *err = rdr2geo_bracket(aztime, slantRange, doppler, orbit, ellipsoid, dem,
                          *targetXYZ, wavelength, side, tolHeight,
                          lookMin, lookMax);
}

// Host geo->radar to test underlying functions in a single-threaded context
CUDA_HOST
int rdr2geo_bracket_h(double aztime, double slantRange, double doppler,
              const isce3::core::Orbit& orbit,
              const isce3::core::Ellipsoid& ellipsoid,
              const isce3::geometry::DEMInterpolator& dem,
              Vec3& targetXYZ, double wavelength, LookSide side,
              double tolHeight, double lookMin, double lookMax)
{
    // Make GPU objects
    isce3::cuda::core::Orbit orbit_d(orbit);
    isce3::cuda::geometry::gpuDEMInterpolator dem_d(dem);

    // Allocate necessary device memory
    int* err_d;
    Vec3* xyz_d;
    checkCudaErrors(cudaMalloc((int**) &err_d, sizeof(int)));
    checkCudaErrors(cudaMalloc((double**) &xyz_d, 3 * sizeof(double)));

    // Run rdr2geo on the GPU
    dim3 grid(1), block(1);
    rdr2geo_bracket_d<<<grid, block>>>(aztime, slantRange, doppler, orbit_d,
                                       ellipsoid, dem_d, xyz_d, wavelength,
                                       side, tolHeight, lookMin, lookMax,
                                       err_d);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaStreamSynchronize(0));

    // Copy results to CPU and return any error code
    int err;
    checkCudaErrors(
            cudaMemcpy(&err, err_d, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(targetXYZ.data(), xyz_d, 3 * sizeof(double),
            cudaMemcpyDeviceToHost));

    // Free memory
    checkCudaErrors(cudaFree(err_d));
    checkCudaErrors(cudaFree(xyz_d));

    // Return error code
    return err;
}
}}} // namespace isce3::cuda::geometry
