#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdio>
#include <fstream>
#include <complex>
#include <ctime>
#include <cstring>

#include "isce/core/Constants.h"
#include "isce/core/DateTime.h"
#include "isce/core/Ellipsoid.h"
#include "isce/core/Peg.h"
#include "isce/core/Pegtrans.h"
#include "isce/geometry/RTC.h"
#include "isce/geometry/Topo.h"

#include "../helper_cuda.h"
#include "isce/cuda/core/gpuEllipsoid.h"
#include "isce/cuda/core/gpuLinAlg.h"
#include "isce/cuda/core/gpuLUT1d.h"
#include "isce/cuda/core/gpuOrbit.h"
#include "isce/cuda/geometry/gpuGeometry.h"
#include "isce/cuda/geometry/gpuDEMInterpolator.h"
#include "isce/cuda/product/gpuImageMode.h"

__constant__ double start, r0, pixazm, dr;
__constant__ float xbound, ybound;

// Function to compute normal vector to a plane given three points
__device__
void computePlaneNormal(double* nhat, const double* p1,
                                      const double* p2,
                                      const double* p3) {
    double p12[3];
    double p13[3];
    double n[3];

    for (int i = 0; i < 3; i++) {
        p12[i] = p2[i] - p1[i];
        p13[i] = p3[i] - p1[i];
    }

    isce::cuda::core::gpuLinAlg::cross(&p13[0], &p12[0], &n[0]);
    isce::cuda::core::gpuLinAlg::unitVec(&n[0], &nhat[0]);
}

__global__ void facet(float* out, size_t xmax, size_t ymax, float upsample_factor,
        isce::cuda::geometry::gpuDEMInterpolator dem_interp,
        isce::cuda::core::gpuEllipsoid ellps,
        isce::cuda::core::gpuOrbit orbit,
        isce::cuda::core::gpuLUT1d<double> dop,
        isce::cuda::product::gpuImageMode mode
        ) {

    const double RAD = M_PI / 180.;

    size_t xidx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t yidx = threadIdx.y + blockIdx.y * blockDim.y;

    // Current latitude
    const double lat0 = dem_interp.yStart() + yidx * dem_interp.deltaY() / upsample_factor;
    const double lat1 = lat0 + dem_interp.deltaY() / upsample_factor;
    const double lat_mid = dem_interp.yStart() + (0.5 + yidx) * dem_interp.deltaY() / upsample_factor;

    using cartesian_t = double[3];
    cartesian_t xyz00, xyz01, xyz10, xyz11, xyz_mid,
                P00_01, P00_10, P10_01, P11_01, P11_10,
                lookXYZ, normalFacet1, normalFacet2;

    const double lon_mid = dem_interp.xStart() + dem_interp.deltaX() * (xidx + 0.5) / upsample_factor;

    double a, r;
    double inputLLH[3] = {lon_mid*RAD, lat_mid*RAD,
        dem_interp.interpolateXY(lon_mid, lat_mid)};

    isce::cuda::geometry::geo2rdr(&inputLLH[0], ellps, orbit, dop, mode,
            &a, &r, 1e-4, 100, 1e-4);

    const float azpix = (a - start) / pixazm;
    const float ranpix = (r - r0) / dr;

    // Establish bounds for bilinear weighting model
    const float x1 = std::floor(ranpix);
    const float x2 = x1 + 1.;
    const float y1 = std::floor(azpix);
    const float y2 = y1 + 1.;

    // Check to see if pixel lies in valid RDC range
    if (ranpix < 0.0 or x2 > xbound or azpix < 0.0 or y2 > ybound)
        return;

    // Current longitude
    const double lon0 = dem_interp.xStart() + dem_interp.deltaX() * xidx / upsample_factor;
    const double lon1 = lon0 + dem_interp.deltaX() / upsample_factor;

    // Set LLH vectors
    double llh00[] = {RAD*lon0, RAD*lat0,
        dem_interp.interpolateXY(lon0, lat0)};
    double llh01[] = {RAD*lon0, RAD*lat1,
        dem_interp.interpolateXY(lon0, lat1)};
    double llh10[] = {RAD*lon1, RAD*lat0,
        dem_interp.interpolateXY(lon1, lat0)};
    double llh11[] = {RAD*lon1, RAD*lat1,
        dem_interp.interpolateXY(lon1, lat1)};

    // Convert to XYZ
    ellps.lonLatToXyz(&llh00[0], xyz00);
    ellps.lonLatToXyz(&llh01[0], xyz01);
    ellps.lonLatToXyz(&llh10[0], xyz10);
    ellps.lonLatToXyz(&llh11[0], xyz11);

    // Compute normal vectors for each facet
    computePlaneNormal(normalFacet1, xyz00, xyz10, xyz01);
    computePlaneNormal(normalFacet2, xyz01, xyz10, xyz11);

    using isce::cuda::core::gpuLinAlg;

    // Compute vectors associated with facet sides
    gpuLinAlg::linComb(1., xyz00, -1., xyz01, P00_01);
    gpuLinAlg::linComb(1., xyz00, -1., xyz10, P00_10);
    gpuLinAlg::linComb(1., xyz10, -1., xyz01, P10_01);
    gpuLinAlg::linComb(1., xyz11, -1., xyz01, P11_01);
    gpuLinAlg::linComb(1., xyz11, -1., xyz10, P11_10);

    // Side lengths
    const double p00_01 = gpuLinAlg::norm(P00_01);
    const double p00_10 = gpuLinAlg::norm(P00_10);
    const double p10_01 = gpuLinAlg::norm(P10_01);
    const double p11_01 = gpuLinAlg::norm(P11_01);
    const double p11_10 = gpuLinAlg::norm(P11_10);

    // Semi-perimeters
    const float h1 = 0.5 * (p00_01 + p00_10 + p10_01);
    const float h2 = 0.5 * (p11_01 + p11_10 + p10_01);

    // Heron's formula to get area of facets in XYZ coordinates
    const float AP1 = std::sqrt(h1 * (h1 - p00_01) * (h1 - p00_10) * (h1 - p10_01));
    const float AP2 = std::sqrt(h2 * (h2 - p11_01) * (h2 - p11_10) * (h2 - p10_01));

    // Compute look angle from sensor to ground
    ellps.lonLatToXyz(inputLLH, xyz_mid);
    double xyz_plat[3];
    double vel[3];
    orbit.interpolateWGS84Orbit(a, &xyz_plat[0], &vel[0]);
    lookXYZ[0] = xyz_plat[0] - xyz_mid[0];
    lookXYZ[1] = xyz_plat[1] - xyz_mid[1];
    lookXYZ[2] = xyz_plat[2] - xyz_mid[2];

    double norm = gpuLinAlg::norm(lookXYZ);
    lookXYZ[0] /= norm;
    lookXYZ[1] /= norm;
    lookXYZ[2] /= norm;

    // Compute dot product between each facet and look vector
    const double cosIncFacet1 = gpuLinAlg::dot(lookXYZ, normalFacet1);
    const double cosIncFacet2 = gpuLinAlg::dot(lookXYZ, normalFacet2);

    // If facets are not illuminated by radar, skip
    if (cosIncFacet1 < 0. or cosIncFacet2 < 0.)
        return;

    // Compute projected area
    const float area = AP1 * cosIncFacet1 + AP2 * cosIncFacet2;

    // Get integer indices of bounds
    const int ix1 = static_cast<int>(x1);
    const int ix2 = static_cast<int>(x2);
    const int iy1 = static_cast<int>(y1);
    const int iy2 = static_cast<int>(y2);

    // Compute fractional weights from indices
    const float Wr = ranpix - x1;
    const float Wa = azpix - y1;
    const float Wrc = 1. - Wr;
    const float Wac = 1. - Wa;

    // Use bilinear weighting to distribute area
    atomicAdd(&out[mode.width() * iy1 + ix1], area * Wrc * Wac);
    atomicAdd(&out[mode.width() * iy1 + ix2], area * Wr  * Wac);
    atomicAdd(&out[mode.width() * iy2 + ix1], area * Wrc * Wa);
    atomicAdd(&out[mode.width() * iy2 + ix2], area * Wr  * Wa);
}

// Compute the flat earth incidence angle correction applied by UAVSAR processing
__global__ void flatearth(float* out,
        const isce::cuda::geometry::gpuDEMInterpolator flat_interp,
        const isce::cuda::core::gpuOrbit orbit,
        const isce::cuda::core::gpuEllipsoid ellps,
        const isce::cuda::product::gpuImageMode mode,
        float lookDir,
        float avg_hgt
        ) {
    size_t j = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i = threadIdx.y + blockIdx.y * blockDim.y;

    if (j >= mode.width() or i >= mode.length())
        return;

    double xyz_plat[3];
    double vel[3];
    orbit.interpolateWGS84Orbit(start + i * pixazm, &xyz_plat[0], &vel[0]);

    // Slant range for current pixel
    const double slt_range = r0 + j * dr;

    // Get LLH and XYZ coordinates for this azimuth/range
    double targetLLH[3];
    double targetXYZ[3];
    targetLLH[2] = avg_hgt; // initialize first guess
    isce::cuda::geometry::rdr2geo(start + i * pixazm, slt_range, 0, orbit, ellps,
            flat_interp, targetLLH, mode.wavelength(), 1,
            1e-4, 20, 20);

    // Computation of ENU coordinates around ground target
    double satToGround[3];
    double enu[3];
    double enumat[9], xyz2enu[9];
    ellps.lonLatToXyz(targetLLH, targetXYZ);
    using isce::cuda::core::gpuLinAlg;
    gpuLinAlg::linComb(1., targetXYZ, -1., xyz_plat, satToGround);
    gpuLinAlg::enuBasis(targetLLH[1], targetLLH[0], enumat);
    gpuLinAlg::tranMat(enumat, xyz2enu);
    gpuLinAlg::matVec(xyz2enu, satToGround, enu);

    // Compute incidence angle components
    const double costheta = fabs(enu[2]) / gpuLinAlg::norm(enu);
    const double sintheta = sqrt(1. - costheta*costheta);

    out[mode.width() * i + j] *= sintheta;
}

double computeUpsamplingFactor(const isce::geometry::DEMInterpolator& dem_interp,
                               const isce::product::ImageMode& mode,
                               const isce::core::Ellipsoid& ellps) {
    // Create a projection object from the DEM interpolator
    isce::core::ProjectionBase * proj = isce::core::createProj(dem_interp.epsgCode());

    // Get middle XY coordinate in DEM coords, lat/lon, and ECEF XYZ
    isce::core::cartesian_t demXY{dem_interp.midX(), dem_interp.midY(), 0.}, llh;
    proj->inverse(demXY, llh);
    isce::core::cartesian_t xyz0;
    ellps.lonLatToXyz(llh, xyz0);

    // Repeat for middle coordinate + deltaX
    demXY[0] += dem_interp.deltaX();
    proj->inverse(demXY, llh);
    isce::core::cartesian_t xyz1;
    ellps.lonLatToXyz(llh, xyz1);

    // Repeat for middle coordinate + deltaX + deltaY
    demXY[1] += dem_interp.deltaY();
    proj->inverse(demXY, llh);
    isce::core::cartesian_t xyz2;
    ellps.lonLatToXyz(llh, xyz2);

    // Estimate width of DEM pixel
    isce::core::cartesian_t delta;
    isce::core::LinAlg::linComb(1., xyz1, -1., xyz0, delta);
    const double dx = isce::core::LinAlg::norm(delta);

    // Estimate length of DEM pixel
    isce::core::LinAlg::linComb(1., xyz2, -1., xyz1, delta);
    const double dy = isce::core::LinAlg::norm(delta);

    // Compute area of DEM pixel
    const double demArea = dx * dy;

    // Compute area of radar pixel (for now, just use spacing in range direction)
    const double radarArea = mode.rangePixelSpacing() * mode.rangePixelSpacing();

    // Upsampling factor is the ratio
    return std::sqrt(demArea / radarArea);
}

template<typename T>
T* deviceCopy(T& host_obj) {
    T* dev_obj;
    checkCudaErrors(cudaMalloc(&dev_obj, sizeof(T)));
    checkCudaErrors(cudaMemcpy(dev_obj, &host_obj, sizeof(T), cudaMemcpyHostToDevice));
    return dev_obj;
}

template<typename T>
T* deviceCopy(T* host_obj) {
    T* dev_obj;
    checkCudaErrors(cudaMalloc(&dev_obj, sizeof(T)));
    checkCudaErrors(cudaMemcpy(dev_obj, host_obj, sizeof(T), cudaMemcpyHostToDevice));
    return dev_obj;
}

namespace isce { namespace cuda {

    namespace core {
        using cartesian_t = double[3];
    };

    namespace geometry {

        void facetRTC(isce::product::Product& product,
                      isce::io::Raster& dem,
                      isce::io::Raster& out_raster) {

            isce::core::Ellipsoid ellps_h = product.metadata().identification().ellipsoid();
            isce::core::Orbit orbit_h(product.metadata().orbitPOE());
            isce::product::ImageMode mode_h = product.complexImagery().primaryMode();
            isce::geometry::Topo topo_h(product);
            topo_h.orbitMethod(isce::core::orbitInterpMethod::HERMITE_METHOD);

            // Initialize other ISCE objects
            isce::core::Peg peg;
            isce::core::Pegtrans ptm;
            ptm.radarToXYZ(ellps_h, peg);

            const double start_h = mode_h.startAzTime().secondsSinceEpoch();
            const double   end   = mode_h.  endAzTime().secondsSinceEpoch();
            const double pixazm_h = (end - start_h) * // azimuth difference per pixel
                    mode_h.numberAzimuthLooks() / mode_h.length();
            const double r0_h = mode_h.startingRange();
            const double dr_h = mode_h.rangePixelSpacing() * mode_h.numberRangeLooks();
            const float xbound_h = mode_h.width()  - 1.;
            const float ybound_h = mode_h.length() - 1.;
            checkCudaErrors(cudaMemcpyToSymbol(start,  &start_h, sizeof(start_h)));
            checkCudaErrors(cudaMemcpyToSymbol(pixazm, &pixazm_h, sizeof(pixazm_h)));
            checkCudaErrors(cudaMemcpyToSymbol(r0, &r0_h, sizeof(r0_h)));
            checkCudaErrors(cudaMemcpyToSymbol(dr, &dr_h, sizeof(dr_h)));
            checkCudaErrors(cudaMemcpyToSymbol(xbound, &xbound_h, sizeof(xbound_h)));
            checkCudaErrors(cudaMemcpyToSymbol(ybound, &ybound_h, sizeof(ybound_h)));

            // Output raster
            auto out = std::make_unique<float[]>(mode_h.length() * mode_h.width());
            float* out_d;
            checkCudaErrors(cudaMalloc(&out_d, mode_h.length() * mode_h.width() * sizeof(float)));

            // ------------------------------------------------------------------------
            // Main code: decompose DEM into facets, compute RDC coordinates
            // ------------------------------------------------------------------------

            // Create CPU-only  objects
            isce::geometry::DEMInterpolator dem_interp_h(0, isce::core::dataInterpMethod::BIQUINTIC_METHOD);
            topo_h.computeDEMBounds(dem, dem_interp_h, 0, mode_h.length()); // determine DEM bounds

            const float upsample_factor = computeUpsamplingFactor(dem_interp_h, mode_h, ellps_h);

            float max_hgt, avg_hgt;
            pyre::journal::info_t info("gpuRTC");
            dem_interp_h.computeHeightStats(max_hgt, avg_hgt, info);
            isce::cuda::geometry::gpuDEMInterpolator flat_interp(avg_hgt);

            // Create hostside device objects
            isce::cuda::geometry::gpuDEMInterpolator dem_interp(dem_interp_h);
            isce::cuda::core::gpuEllipsoid ellps(ellps_h);
            isce::cuda::core::gpuOrbit orbit(orbit_h);
            isce::cuda::core::gpuLUT1d<double> dop(product.metadata().instrument().skewDoppler());
            isce::cuda::product::gpuImageMode mode(mode_h);

            const size_t xmax = dem_interp.width()  * upsample_factor;
            const size_t ymax = dem_interp.length() * upsample_factor;

            dem_interp.initProjInterp();

#define BLOCK_X 16
#define BLOCK_Y 16
            static_assert(BLOCK_X * BLOCK_Y <= 256,
                    "RTC block dim too large for resources available on GPU");

            {
                dim3 block(BLOCK_X, BLOCK_Y);
                dim3 grid(xmax / BLOCK_X + 1,
                          ymax / BLOCK_Y + 1);
                facet<<<grid, block>>>(out_d, xmax, ymax, upsample_factor,
                                       dem_interp, ellps, orbit, dop, mode);
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
            }

            {
                dim3 block(BLOCK_X, BLOCK_Y);
                dim3 grid(mode.width() / BLOCK_X + 1,
                          mode.width() / BLOCK_Y + 1);
                flatearth<<<grid, block>>>(out_d, flat_interp, orbit, ellps, mode,
                        product.metadata().identification().lookDirection(), avg_hgt);
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
            }

            checkCudaErrors(cudaMemcpy(&out[0], out_d, mode.width() * mode.length() * sizeof(float),
                                       cudaMemcpyDeviceToHost));
            out_raster.setBlock(&out[0], 0, 0, mode.width(), mode.length());
        }
    };

}; };
