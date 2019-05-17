//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright: 2017-2018

#include <isce/core/Basis.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Pixel.h>

// isce::cuda::core
#include <isce/cuda/core/gpuOrbit.h>
#include <isce/cuda/core/gpuLUT1d.h>
#include <isce/cuda/core/gpuStateVector.h>

// isce::cuda::geometry
#include "gpuDEMInterpolator.h"
#include "gpuGeometry.h"
#include "gpuTopoLayers.h"
#include "gpuTopo.h"

using isce::core::LinAlg;
using isce::core::Vec3;

#include <isce/cuda/except/Error.h>

#define THRD_PER_BLOCK 96 // Number of threads per block (should always %32==0)

__device__
bool initAzimuthLine(size_t line,
                     const isce::cuda::core::gpuOrbit& orbit,
                     double startAzUTCTime,
                     size_t numberAzimuthLooks,
                     double prf,
                     isce::cuda::core::gpuStateVector& state,
                     isce::core::Basis& TCNbasis) {

    // Get satellite azimuth time
    const double tline = startAzUTCTime + numberAzimuthLooks * line / prf;

    // Interpolate orbit (keeping track of validity without interrupting workflow)
    Vec3 pos, vel;
    bool valid = (orbit.interpolateWGS84Orbit(tline, pos.data(), vel.data()) == 0);
    state._position = pos;
    state._velocity = vel;

    // Compute geocentric TCN basis
    const Vec3 nhat = -pos.unitVec();
    const Vec3 chat = nhat.cross(vel ).unitVec();
    const Vec3 that = chat.cross(nhat).unitVec();

    TCNbasis = isce::core::Basis(that, chat, nhat);

    return valid;
}

__device__
void setOutputTopoLayers(const Vec3& targetLLH,
                         isce::cuda::geometry::gpuTopoLayers & layers,
                         size_t index, int lookSide,
                         const isce::core::Pixel & pixel,
                         const isce::cuda::core::gpuStateVector & state,
                         const isce::core::Basis& TCNbasis,
                         isce::cuda::core::ProjectionBase ** projOutput,
                         const isce::core::Ellipsoid& ellipsoid,
                         const isce::cuda::geometry::gpuDEMInterpolator & demInterp) {

    Vec3 targetXYZ, enu;
    isce::core::cartmat_t enumat, xyz2enu;
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
    const Vec3 satToGround = targetXYZ - state.position();

    // Compute cross-track range
    layers.crossTrack(index, -lookSide * satToGround.dot(TCNbasis.x1()));

    // Computation in ENU coordinates around target
    LinAlg::enuBasis(targetLLH[1], targetLLH[0], enumat);
    LinAlg::tranMat(enumat, xyz2enu);
    LinAlg::matVec(xyz2enu, satToGround, enu);
    const double cosalpha = std::abs(enu[2]) / LinAlg::norm(enu);

    // LOS vectors
    layers.inc(index, std::acos(cosalpha) * degrees);
    layers.hdg(index, (std::atan2(-enu[1], -enu[0]) - (0.5*M_PI)) * degrees);

    // East-west slope using central difference
    double aa = demInterp.interpolateXY(x - demInterp.deltaX(), y);
    double bb = demInterp.interpolateXY(x + demInterp.deltaX(), y);
    double gamma = targetLLH[1];
    double alpha = ((bb - aa) * degrees) / (2.0 * ellipsoid.rEast(gamma) * demInterp.deltaX());

    // North-south slope using central difference
    aa = demInterp.interpolateXY(x, y - demInterp.deltaY());
    bb = demInterp.interpolateXY(x, y + demInterp.deltaY());
    double beta = ((bb - aa) * degrees) / (2.0 * ellipsoid.rNorth(gamma) * demInterp.deltaY());

    // Compute local incidence angle
    const double enunorm = LinAlg::norm(enu);
    for (int idx = 0; idx < 3; ++idx) {
        enu[idx] = enu[idx] / enunorm;
    }
    double costheta = ((enu[0] * alpha) + (enu[1] * beta) - enu[2])
                     / std::sqrt(1.0 + (alpha * alpha) + (beta * beta));
    layers.localInc(index, std::acos(costheta)*degrees);

    // Compute amplitude simulation
    double sintheta = std::sqrt(1.0 - (costheta * costheta));
    bb = sintheta + 0.1 * costheta;
    layers.sim(index, std::log10(std::abs(0.01 * costheta / (bb * bb * bb))));

    // Calculate psi angle between image plane and local slope
    Vec3 n_img_enu, n_trg_enu;
    const Vec3 n_imghat = satToGround.cross(state.velocity()).unitVec() * -lookSide;
    LinAlg::matVec(xyz2enu, n_imghat, n_img_enu);
    n_trg_enu[0] = -alpha;
    n_trg_enu[1] = -beta;
    n_trg_enu[2] = 1.0;

    const double cospsi = LinAlg::dot(n_trg_enu, n_img_enu) /
                         (LinAlg::norm(n_trg_enu) * LinAlg::norm(n_img_enu));
    layers.localPsi(index, std::acos(cospsi) * degrees);
}

__global__
void runTopoBlock(isce::core::Ellipsoid ellipsoid,
                  isce::cuda::core::gpuOrbit orbit,
                  isce::cuda::core::gpuLUT1d<double> doppler,
                  isce::cuda::geometry::gpuDEMInterpolator demInterp,
                  isce::cuda::core::ProjectionBase ** projOutput,
                  isce::cuda::geometry::gpuTopoLayers layers,
                  size_t lineStart,
                  int lookSide,
                  size_t numberAzimuthLooks,
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
        isce::cuda::core::gpuStateVector state;
        isce::core::Basis TCNbasis;
        bool valid = (initAzimuthLine(line + lineStart, orbit, startAzUTCTime,
                                      numberAzimuthLooks, prf, state, TCNbasis) != 0);

        // Compute magnitude of satellite velocity
        const double satVmag = state.velocity().norm();

        // Get current slant range
        const double rng = startingRange + rbin * rangePixelSpacing;

        // Get current Doppler value and factor
        const double dopval = doppler.eval(rng);
        const double dopfact = 0.5 * wavelength * (dopval / satVmag) * rng;

        // Store slant range bin data in Pixel
        isce::core::Pixel pixel(rng, dopfact, rbin);

        // Initialize LLH to middle of input DEM and average height
        Vec3 llh = demInterp.midLonLat();

        // Perform rdr->geo iterations
        int geostat = isce::cuda::geometry::rdr2geo(
            pixel, TCNbasis, state, ellipsoid, demInterp, llh, lookSide,
            threshold, numiter, extraiter);

        // Save data in output arrays
        setOutputTopoLayers(llh, layers, index_flat, lookSide, pixel, state, TCNbasis,
                            projOutput, ellipsoid, demInterp);

        // Update convergence count
        atomicAdd(totalconv, (unsigned int) geostat);
    }
}

// C++ Host code for launching kernel to run topo on current block
void isce::cuda::geometry::
runGPUTopo(const isce::core::Ellipsoid & ellipsoid,
           const isce::core::Orbit & orbit,
           const isce::core::LUT1d<double> & doppler,
           isce::geometry::DEMInterpolator & demInterp,
           isce::geometry::TopoLayers & layers,
           size_t lineStart,
           int lookSide,
           int epsgOut,
           size_t numberAzimuthLooks,
           double startAzUTCTime,
           double wavelength,
           double prf,
           double startingRange,
           double rangePixelSpacing,
           double threshold, int numiter, int extraiter,
           unsigned int & totalconv) {

    // Create gpu ISCE objects
    isce::cuda::core::gpuOrbit gpu_orbit(orbit);
    isce::cuda::core::gpuLUT1d<double> gpu_doppler(doppler);
    isce::cuda::geometry::gpuDEMInterpolator gpu_demInterp(demInterp); 
    isce::cuda::geometry::gpuTopoLayers gpu_layers(layers);
    
    // Allocate projection pointers on device
    isce::cuda::core::ProjectionBase **projOutput_d;
    checkCudaErrors(cudaMalloc(&projOutput_d, sizeof(isce::cuda::core::ProjectionBase **)));
    createProjection<<<1, 1>>>(projOutput_d, epsgOut);

    // DEM interpolator initializes its projection and interpolator
    gpu_demInterp.initProjInterp();

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
                                  lineStart, lookSide, numberAzimuthLooks,
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
    gpu_demInterp.finalizeProjInterp();
    deleteProjection<<<1, 1>>>(projOutput_d);

    // Free projection pointer and convergence count
    checkCudaErrors(cudaFree(totalconv_d));
    checkCudaErrors(cudaFree(projOutput_d));
}

// end of file
