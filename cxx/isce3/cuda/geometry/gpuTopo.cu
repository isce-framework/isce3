#include "gpuTopo.h"
#include <cmath>

#include <isce3/core/Basis.h>
#include <isce3/core/DenseMatrix.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/Pixel.h>
#include <isce3/error/ErrorCode.h>
#include <isce3/geometry/TopoLayers.h>

// isce3::cuda::core
#include <isce3/cuda/core/Orbit.h>
#include <isce3/cuda/core/OrbitView.h>
#include <isce3/cuda/core/gpuLUT1d.h>

#include <isce3/cuda/except/Error.h>

// isce3::cuda::geometry
#include "gpuDEMInterpolator.h"
#include "gpuGeometry.h"
#include "gpuTopoLayers.h"

using isce3::core::Vec3;
using isce3::core::Mat3;
using isce3::error::ErrorCode;
using isce3::core::LookSide;

#define THRD_PER_BLOCK 96 // Number of threads per block (should always %32==0)

__device__
bool initAzimuthLine(size_t line,
                     const isce3::cuda::core::OrbitView& orbit,
                     double startAzUTCTime,
                     double prf,
                     Vec3& pos, Vec3& vel,
                     isce3::core::Basis& TCNbasis) {

    // Get satellite azimuth time
    const double tline = startAzUTCTime + line / prf;

    // Interpolate orbit (keeping track of validity without interrupting workflow)
    ErrorCode status = orbit.interpolate(&pos, &vel, tline);
    bool valid = (status == ErrorCode::Success);

    // Compute geocentric TCN basis
    TCNbasis = isce3::core::Basis(pos, vel);

    return valid;
}

__device__
void setOutputTopoLayers(const Vec3& targetLLH,
                         isce3::cuda::geometry::gpuTopoLayers & layers,
                         size_t index, LookSide lookSide,
                         const isce3::core::Pixel & pixel,
                         const Vec3& pos, const Vec3& vel,
                         const isce3::core::Basis& TCNbasis,
                         isce3::cuda::core::ProjectionBase ** projOutput,
                         const isce3::core::Ellipsoid& ellipsoid,
                         const isce3::cuda::geometry::gpuDEMInterpolator & demInterp) {

    Vec3 targetXYZ, enu;
    const double degrees = 180.0 / M_PI;

    // Convert lat/lon values to output coordinate system
    Vec3 xyzOut;
    (*projOutput)->forward(targetLLH, xyzOut);
    const double x = xyzOut[0];
    const double y = xyzOut[1];

    // Set outputs
    layers.x(index, x);
    layers.y(index, y);
    layers.z(index, targetLLH[2]);

    // Convert llh->xyz for ground point
    ellipsoid.lonLatToXyz(targetLLH, targetXYZ);

    // Compute vector from satellite to ground point
    const Vec3 satToGround = targetXYZ - pos;

    // Compute cross-track range
    if (lookSide == LookSide::Left) {
        layers.crossTrack(index, -satToGround.dot(TCNbasis.x1()));
    } else {
        layers.crossTrack(index, satToGround.dot(TCNbasis.x1()));
    }

    // Computation in ENU coordinates around target
    const Mat3 xyz2enu = Mat3::xyzToEnu(targetLLH[1], targetLLH[0]);
    enu = xyz2enu.dot(satToGround);
    const double cosalpha = std::abs(enu[2]) / enu.norm();

    // Incidence angle
    layers.inc(index, std::acos(cosalpha) * degrees);

    // Heading considering zero-Doppler grid and anti-clock. ref. starting from the East
    double heading;
    if (lookSide == LookSide::Left) {
        heading = (std::atan2(enu[1], enu[0]) - (0.5*M_PI)) * degrees;
    } else {
        heading = (std::atan2(enu[1], enu[0]) + (0.5*M_PI)) * degrees;
    }

    if (heading > 180) {
        heading -= 360;
    } else if (heading < -180) {
        heading += 360;
    }
    layers.hdg(index, heading);

    // Compute and assign ground to satellite unit vector east and north components
    const Vec3 groundToSat = -satToGround;
    const Vec3 enuGroundToSat = xyz2enu.dot(groundToSat).normalized();
    layers.groundToSatEast(index, enuGroundToSat[0]);
    layers.groundToSatNorth(index, enuGroundToSat[1]);

    // Project output coordinates to DEM coordinates
    Vec3 input_coords_llh;
    (*projOutput)->inverse({x, y, targetLLH[2]}, input_coords_llh);
    Vec3 dem_vect;
    (*(demInterp.proj()))->forward(input_coords_llh, dem_vect);

    // East-west slope using central difference
    double aa = demInterp.interpolateXY(dem_vect[0] - demInterp.deltaX(), dem_vect[1]);
    double bb = demInterp.interpolateXY(dem_vect[0] + demInterp.deltaX(), dem_vect[1]);

    Vec3 dem_vect_p_dx = {dem_vect[0] + demInterp.deltaX(), dem_vect[1], dem_vect[2]};
    Vec3 dem_vect_m_dx = {dem_vect[0] - demInterp.deltaX(), dem_vect[1], dem_vect[2]};
    Vec3 input_coords_llh_p_dx, input_coords_llh_m_dx;
    (*(demInterp.proj()))->inverse(dem_vect_p_dx, input_coords_llh_p_dx);
    (*(demInterp.proj()))->inverse(dem_vect_m_dx, input_coords_llh_m_dx);
    const Vec3 input_coords_xyz_p_dx = ellipsoid.lonLatToXyz(input_coords_llh_p_dx);
    const Vec3 input_coords_xyz_m_dx = ellipsoid.lonLatToXyz(input_coords_llh_m_dx);
    double dx = (input_coords_xyz_p_dx - input_coords_xyz_m_dx).norm();

    // Compute east-west slope using plus-minus sign from deltaX() (usually positive)
    double alpha = std::copysign((bb - aa) / dx, (bb - aa) * demInterp.deltaX());

    // North-south slope using central difference
    aa = demInterp.interpolateXY(dem_vect[0], dem_vect[1] - demInterp.deltaY());
    bb = demInterp.interpolateXY(dem_vect[0], dem_vect[1] + demInterp.deltaY());

    Vec3 dem_vect_p_dy = {dem_vect[0], dem_vect[1] + demInterp.deltaY(), dem_vect[2]};
    Vec3 dem_vect_m_dy = {dem_vect[0], dem_vect[1] - demInterp.deltaY(), dem_vect[2]};
    Vec3 input_coords_llh_p_dy, input_coords_llh_m_dy;
    (*(demInterp.proj()))->inverse(dem_vect_p_dy, input_coords_llh_p_dy);
    (*(demInterp.proj()))->inverse(dem_vect_m_dy, input_coords_llh_m_dy);
    const Vec3 input_coords_xyz_p_dy = ellipsoid.lonLatToXyz(input_coords_llh_p_dy);
    const Vec3 input_coords_xyz_m_dy = ellipsoid.lonLatToXyz(input_coords_llh_m_dy);
    double dy = (input_coords_xyz_p_dy - input_coords_xyz_m_dy).norm();

    // Compute north-south slope using plus-minus sign from deltaY() (usually negative)
    double beta = std::copysign((bb - aa) / dy, (bb - aa) * demInterp.deltaY());

    // Compute local incidence angle
    enu /= enu.norm();
    double costheta = ((enu[0] * alpha) + (enu[1] * beta) - enu[2])
                     / std::sqrt(1.0 + (alpha * alpha) + (beta * beta));
    layers.localInc(index, std::acos(costheta)*degrees);

    // Compute amplitude simulation
    double sintheta = std::sqrt(1.0 - (costheta * costheta));
    bb = sintheta + 0.1 * costheta;
    layers.sim(index, std::log10(std::abs(0.01 * costheta / (bb * bb * bb))));

    // Calculate psi angle between image plane and local slope
    Vec3 n_img_enu, n_trg_enu;
    Vec3 n_imghat = satToGround.cross(vel).normalized();
    if (lookSide == LookSide::Left) {
        n_imghat *= -1;
    }
    n_img_enu = xyz2enu.dot(n_imghat);
    n_trg_enu[0] = -alpha;
    n_trg_enu[1] = -beta;
    n_trg_enu[2] = 1.0;

    const double cospsi = n_img_enu.dot(n_trg_enu) /
                         (n_trg_enu.norm() * n_img_enu.norm());
    layers.localPsi(index, std::acos(cospsi) * degrees);
}

__global__
void runTopoBlock(isce3::core::Ellipsoid ellipsoid,
                  isce3::cuda::core::OrbitView orbit,
                  isce3::cuda::core::gpuLUT1d<double> doppler,
                  isce3::cuda::geometry::gpuDEMInterpolator demInterp,
                  isce3::cuda::core::ProjectionBase ** projOutput,
                  isce3::cuda::geometry::gpuTopoLayers layers,
                  size_t lineStart,
                  LookSide lookSide,
                  double startAzUTCTime,
                  double wavelength,
                  double prf,
                  double startingRange,
                  double rangePixelSpacing,
                  double threshold, int numiter, int extraiter,
                  unsigned int * totalconv) {

    // Get the flattened index
    size_t index_flat = (blockDim.x * blockIdx.x) + threadIdx.x;
    const size_t NPIXELS = layers.length() * layers.width();

    // Only process if a valid pixel (while trying to avoid thread divergence)
    if (index_flat < NPIXELS) {

        // Unravel the flattened pixel index
        const size_t line = index_flat / layers.width();
        const size_t rbin = index_flat - line * layers.width();

        // Interpolate orbit (keeping track of validity without interrupting workflow)
        isce3::core::Basis TCNbasis;
        Vec3 pos, vel;
        bool valid = (initAzimuthLine(line + lineStart, orbit, startAzUTCTime,
                                      prf, pos, vel, TCNbasis) != 0);

        // Compute magnitude of satellite velocity
        const double satVmag = vel.norm();

        // Get current slant range
        const double rng = startingRange + rbin * rangePixelSpacing;

        // Get current Doppler value and factor
        const double dopval = doppler.eval(rng);
        const double dopfact = 0.5 * wavelength * (dopval / satVmag) * rng;

        // Store slant range bin data in Pixel
        isce3::core::Pixel pixel(rng, dopfact, rbin);

        // Initialize LLH to middle of input DEM and average height
        Vec3 llh = demInterp.midLonLat();

        // Perform rdr->geo iterations
        int geostat = isce3::cuda::geometry::rdr2geo(
            pixel, TCNbasis, pos, vel, ellipsoid, demInterp, llh, lookSide,
            threshold, numiter, extraiter);

        // Save data in output arrays
        setOutputTopoLayers(llh, layers, index_flat, lookSide, pixel, pos, vel, TCNbasis,
                            projOutput, ellipsoid, demInterp);

        // Update convergence count
        atomicAdd(totalconv, (unsigned int) geostat);
    }
}

// C++ Host code for launching kernel to run topo on current block
void isce3::cuda::geometry::
runGPUTopo(const isce3::core::Ellipsoid & ellipsoid,
           const isce3::core::Orbit & orbit,
           const isce3::core::LUT1d<double> & doppler,
           isce3::geometry::DEMInterpolator & demInterp,
           isce3::geometry::TopoLayers & layers,
           size_t lineStart,
           LookSide lookSide,
           int epsgOut,
           double startAzUTCTime,
           double wavelength,
           double prf,
           double startingRange,
           double rangePixelSpacing,
           double threshold, int numiter, int extraiter,
           unsigned int & totalconv) {

    // Create gpu ISCE objects
    isce3::cuda::core::Orbit gpu_orbit(orbit);
    isce3::cuda::core::gpuLUT1d<double> gpu_doppler(doppler);
    isce3::cuda::geometry::gpuDEMInterpolator gpu_demInterp(demInterp);
    isce3::cuda::geometry::gpuTopoLayers gpu_layers(layers);

    // Allocate projection pointers on device
    isce3::cuda::core::ProjectionBase **projOutput_d;
    checkCudaErrors(cudaMalloc(&projOutput_d, sizeof(isce3::cuda::core::ProjectionBase **)));
    createProjection<<<1, 1>>>(projOutput_d, epsgOut);

    // Allocate integer for storing convergence results
    unsigned int * totalconv_d;
    checkCudaErrors(cudaMalloc(&totalconv_d, sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(totalconv_d, &totalconv, sizeof(unsigned int),
                               cudaMemcpyHostToDevice));

    // Determine grid layout
    dim3 block(THRD_PER_BLOCK);
    const size_t npixel = layers.length() * layers.width();
    const int nBlocks = (int) std::ceil((1.0 * npixel) / THRD_PER_BLOCK);
    dim3 grid(nBlocks);

    // Launch kernel
    runTopoBlock<<<grid, block>>>(ellipsoid, gpu_orbit, gpu_doppler,
                                  gpu_demInterp, projOutput_d, gpu_layers,
                                  lineStart, lookSide,
                                  startAzUTCTime, wavelength, prf, startingRange,
                                  rangePixelSpacing, threshold, numiter, extraiter,
                                  totalconv_d);

    // Check for any kernel errors
    checkCudaErrors(cudaPeekAtLastError());

    // Copy results back to host
    gpu_layers.copyToHost(layers);
    checkCudaErrors(cudaMemcpy(&totalconv, totalconv_d, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));

    // Delete projection pointer on device
    deleteProjection<<<1, 1>>>(projOutput_d);

    // Free projection pointer and convergence count
    checkCudaErrors(cudaFree(totalconv_d));
    checkCudaErrors(cudaFree(projOutput_d));
}
