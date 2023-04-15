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

#include <isce3/core/Constants.h>
#include <isce3/core/DateTime.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/LUT1d.h>
#include <isce3/core/Peg.h>
#include <isce3/core/Pegtrans.h>
#include <isce3/core/Projections.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/RTC.h>
#include <isce3/geometry/Topo.h>
#include <isce3/product/RadarGridProduct.h>

#include <isce3/cuda/core/gpuLUT1d.h>
#include <isce3/cuda/core/Orbit.h>
#include <isce3/cuda/core/OrbitView.h>
#include <isce3/cuda/except/Error.h>
#include <isce3/cuda/geometry/gpuGeometry.h>
#include <isce3/cuda/geometry/gpuDEMInterpolator.h>

__constant__ double start, r0, pixazm, dr;
__constant__ float xbound, ybound;

using isce3::core::avgLUT2dToLUT1d;
using isce3::core::OrbitInterpBorderMode;
using isce3::core::Vec3;
using isce3::core::Mat3;

__global__ void facet(float* out, size_t xmax, size_t ymax, float upsample_factor,
        isce3::cuda::geometry::gpuDEMInterpolator dem_interp,
        isce3::core::Ellipsoid ellps,
        isce3::cuda::core::OrbitView orbit,
        isce3::cuda::core::gpuLUT1d<double> dop,
        size_t width,
        double wavelength,
        isce3::core::LookSide side) {

    size_t xidx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t yidx = threadIdx.y + blockIdx.y * blockDim.y;

    // Current y-coord in DEM
    const double dem_y0 = dem_interp.yStart() + yidx * dem_interp.deltaY() / upsample_factor;
    const double dem_y1 = dem_y0 + dem_interp.deltaY() / upsample_factor;
    const double dem_ymid = dem_interp.yStart() + (0.5 + yidx) * dem_interp.deltaY() / upsample_factor;

    Vec3 lookXYZ;

    const double dem_xmid = dem_interp.xStart() + dem_interp.deltaX() * (xidx + 0.5) / upsample_factor;

    double a, r;
    Vec3 inputLLH;
    const Vec3 inputDEM { dem_xmid, dem_ymid,
                          dem_interp.interpolateXY(dem_xmid, dem_ymid) };
    int epsgcode = dem_interp.epsgCode();
    isce3::cuda::core::projInverse(epsgcode, inputDEM, inputLLH);

    isce3::cuda::geometry::geo2rdr(inputLLH, ellps, orbit, dop,
                                  &a, &r, wavelength, side,
                                  1e-4, 20, 1e-4);

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

    // Current x-coord in DEM
    const double dem_x0 = dem_interp.xStart() + dem_interp.deltaX() * xidx / upsample_factor;
    const double dem_x1 = dem_x0 + dem_interp.deltaX() / upsample_factor;

    // Set DEM-coordinate corner vectors
    const Vec3 dem00 {dem_x0, dem_y0,
        dem_interp.interpolateXY(dem_x0, dem_y0)};
    const Vec3 dem01 {dem_x0, dem_y1,
        dem_interp.interpolateXY(dem_x0, dem_y1)};
    const Vec3 dem10 {dem_x1, dem_y0,
        dem_interp.interpolateXY(dem_x1, dem_y0)};
    const Vec3 dem11 {dem_x1, dem_y1,
        dem_interp.interpolateXY(dem_x1, dem_y1)};

    // Get LLH corner vectors
    Vec3 llh00, llh01, llh10, llh11;
    isce3::cuda::core::projInverse(epsgcode, dem00, llh00);
    isce3::cuda::core::projInverse(epsgcode, dem01, llh01);
    isce3::cuda::core::projInverse(epsgcode, dem10, llh10);
    isce3::cuda::core::projInverse(epsgcode, dem11, llh11);

    // Convert to XYZ
    const Vec3 xyz00 = ellps.lonLatToXyz(llh00);
    const Vec3 xyz01 = ellps.lonLatToXyz(llh01);
    const Vec3 xyz10 = ellps.lonLatToXyz(llh10);
    const Vec3 xyz11 = ellps.lonLatToXyz(llh11);

    // Compute normal vectors for each facet
    const Vec3 normalFacet1 = isce3::core::normalPlane(xyz00, xyz10, xyz01);
    const Vec3 normalFacet2 = isce3::core::normalPlane(xyz01, xyz10, xyz11);

    // Side lengths
    const double p00_01 = (xyz00 - xyz01).norm();
    const double p00_10 = (xyz00 - xyz10).norm();
    const double p10_01 = (xyz10 - xyz01).norm();
    const double p11_01 = (xyz11 - xyz01).norm();
    const double p11_10 = (xyz11 - xyz10).norm();

    // Semi-perimeters
    const float h1 = 0.5 * (p00_01 + p00_10 + p10_01);
    const float h2 = 0.5 * (p11_01 + p11_10 + p10_01);

    // Heron's formula to get area of facets in XYZ coordinates
    const float AP1 = std::sqrt(h1 * (h1 - p00_01) * (h1 - p00_10) * (h1 - p10_01));
    const float AP2 = std::sqrt(h2 * (h2 - p11_01) * (h2 - p11_10) * (h2 - p10_01));

    // Compute look angle from sensor to ground
    const Vec3 xyz_mid = ellps.lonLatToXyz(inputLLH);
    Vec3 xyz_plat;
    orbit.interpolate(&xyz_plat, nullptr, a, OrbitInterpBorderMode::FillNaN);
    lookXYZ = (xyz_plat - xyz_mid).normalized();

    // Compute dot product between each facet and look vector
    const double cosIncFacet1 = lookXYZ.dot(normalFacet1);
    const double cosIncFacet2 = lookXYZ.dot(normalFacet2);

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
    atomicAdd(&out[width * iy1 + ix1], area * Wrc * Wac);
    atomicAdd(&out[width * iy1 + ix2], area * Wr  * Wac);
    atomicAdd(&out[width * iy2 + ix1], area * Wrc * Wa);
    atomicAdd(&out[width * iy2 + ix2], area * Wr  * Wa);
}

// Compute the flat earth incidence angle correction applied by UAVSAR processing
__global__ void flatearth(float* out,
        const isce3::cuda::geometry::gpuDEMInterpolator flat_interp,
        const isce3::cuda::core::OrbitView orbit,
        const isce3::core::Ellipsoid ellps,
        size_t length,
        size_t width,
        double wavelength,
        isce3::core::LookSide lookSide,
        float avg_hgt
        ) {
    size_t j = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i = threadIdx.y + blockIdx.y * blockDim.y;

    if (j >= width or i >= length)
        return;

    Vec3 xyz_plat;
    double t = start + i * pixazm;
    orbit.interpolate(&xyz_plat, nullptr, t, OrbitInterpBorderMode::FillNaN);

    // Slant range for current pixel
    const double slt_range = r0 + j * dr;

    // Get LLH and XYZ coordinates for this azimuth/range
    Vec3 targetLLH;
    targetLLH[2] = avg_hgt; // initialize first guess
    isce3::cuda::geometry::rdr2geo(start + i * pixazm, slt_range, 0, orbit, ellps,
            flat_interp, targetLLH, wavelength, lookSide,
            1e-4, 20, 20);

    // Computation of ENU coordinates around ground target
    const Vec3 targetXYZ = ellps.lonLatToXyz(targetLLH);
    const Vec3 satToGround = targetXYZ - xyz_plat;

    const Mat3 xyz2enu = Mat3::xyzToEnu(targetLLH[1], targetLLH[0]);
    const Vec3 enu = xyz2enu.dot(satToGround);

    // Compute incidence angle components
    const double costheta = fabs(enu[2]) / enu.norm();
    const double sintheta = sqrt(1. - costheta*costheta);

    out[width * i + j] *= sintheta;
}

double computeUpsamplingFactor(const isce3::geometry::DEMInterpolator& dem_interp,
                               const isce3::core::Ellipsoid& ellps,
                               double rangePixelSpacing) {
    // Create a projection object from the DEM interpolator
    isce3::core::ProjectionBase * proj = isce3::core::createProj(dem_interp.epsgCode());

    // Get middle XY coordinate in DEM coords, lat/lon, and ECEF XYZ
    Vec3 demXY{dem_interp.midX(), dem_interp.midY(), 0.}, llh;
    proj->inverse(demXY, llh);
    Vec3 xyz0;
    ellps.lonLatToXyz(llh, xyz0);

    // Repeat for middle coordinate + deltaX
    demXY[0] += dem_interp.deltaX();
    proj->inverse(demXY, llh);
    Vec3 xyz1;
    ellps.lonLatToXyz(llh, xyz1);

    // Repeat for middle coordinate + deltaX + deltaY
    demXY[1] += dem_interp.deltaY();
    proj->inverse(demXY, llh);
    Vec3 xyz2;
    ellps.lonLatToXyz(llh, xyz2);

    // Estimate width of DEM pixel
    Vec3 delta = xyz1 - xyz0;
    const double dx = delta.norm();

    // Estimate length of DEM pixel
    delta = xyz2 - xyz1;
    const double dy = delta.norm();

    // Compute area of DEM pixel
    const double demArea = dx * dy;

    // Compute area of radar pixel (for now, just use spacing in range direction)
    const double radarArea = rangePixelSpacing * rangePixelSpacing;

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

namespace isce3 { namespace cuda {

    namespace geometry {

void computeRtc(isce3::product::RadarGridProduct& product, isce3::io::Raster& dem,
                isce3::io::Raster& out_raster, char frequency)
{

    isce3::core::Ellipsoid ellps_h;
    isce3::core::Orbit orbit_h(product.metadata().orbit());
    isce3::product::RadarGridParameters radarGrid(product, frequency);
    isce3::geometry::Topo topo_h(product, frequency, true);
    const isce3::core::LookSide lookDirection = product.lookSide();

    // Initialize other ISCE objects
    isce3::core::Peg peg;
    isce3::core::Pegtrans ptm;
    ptm.radarToXYZ(ellps_h, peg);

    const double start_h = radarGrid.sensingStart();
    const double end = radarGrid.sensingStop();
    const double pixazm_h = (end - start_h) /
                            radarGrid.length(); // azimuth difference per pixel
    const double r0_h = radarGrid.startingRange();
    const double dr_h = radarGrid.rangePixelSpacing();
    const float xbound_h = radarGrid.width() - 1.;
    const float ybound_h = radarGrid.length() - 1.;
    checkCudaErrors(cudaMemcpyToSymbol(start, &start_h, sizeof(start_h)));
    checkCudaErrors(cudaMemcpyToSymbol(pixazm, &pixazm_h, sizeof(pixazm_h)));
    checkCudaErrors(cudaMemcpyToSymbol(r0, &r0_h, sizeof(r0_h)));
    checkCudaErrors(cudaMemcpyToSymbol(dr, &dr_h, sizeof(dr_h)));
    checkCudaErrors(cudaMemcpyToSymbol(xbound, &xbound_h, sizeof(xbound_h)));
    checkCudaErrors(cudaMemcpyToSymbol(ybound, &ybound_h, sizeof(ybound_h)));

    // Output raster
    auto out = std::make_unique<float[]>(radarGrid.size());
    float* out_d;
    checkCudaErrors(cudaMalloc(&out_d, radarGrid.size() * sizeof(float)));

    // ------------------------------------------------------------------------
    // Main code: decompose DEM into facets, compute RDC coordinates
    // ------------------------------------------------------------------------

    // Create CPU-only  objects
    isce3::geometry::DEMInterpolator dem_interp_h(
            0, isce3::core::dataInterpMethod::BIQUINTIC_METHOD);
    topo_h.computeDEMBounds(dem, dem_interp_h, 0,
                            radarGrid.length()); // determine DEM bounds

    const float upsample_factor = computeUpsamplingFactor(
            dem_interp_h, ellps_h, radarGrid.rangePixelSpacing());

    float min_hgt, max_hgt, avg_hgt;
    pyre::journal::info_t info("gpuRTC");
    dem_interp_h.computeMinMaxMeanHeight(min_hgt, max_hgt, avg_hgt);
    isce3::cuda::geometry::gpuDEMInterpolator flat_interp(avg_hgt);

    // Create hostside device objects
    isce3::cuda::geometry::gpuDEMInterpolator dem_interp(dem_interp_h);
    isce3::core::Ellipsoid ellps(ellps_h);
    isce3::cuda::core::Orbit orbit(orbit_h);

    // Convert LUT2d doppler to LUT1d
    isce3::core::LUT1d<double> dop_h(
            avgLUT2dToLUT1d(product.metadata().procInfo().dopplerCentroid(frequency)));
    isce3::cuda::core::gpuLUT1d<double> dop(dop_h);

    const size_t xmax = dem_interp.width() * upsample_factor;
    const size_t ymax = dem_interp.length() * upsample_factor;

#define BLOCK_X 16
#define BLOCK_Y 16
            static_assert(BLOCK_X * BLOCK_Y <= 256,
                    "RTC block dim too large for resources available on GPU");

            {
                dim3 block(BLOCK_X, BLOCK_Y);
                dim3 grid(xmax / BLOCK_X + 1,
                          ymax / BLOCK_Y + 1);
                facet<<<grid, block>>>(out_d, xmax, ymax, upsample_factor,
                                       dem_interp, ellps, orbit, dop,
                                       radarGrid.width(), radarGrid.wavelength(),
                                       lookDirection);
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
            }

            {
                dim3 block(BLOCK_X, BLOCK_Y);
                dim3 grid(radarGrid.width() / BLOCK_X + 1,
                          radarGrid.width() / BLOCK_Y + 1);
                flatearth<<<grid, block>>>(out_d, flat_interp, orbit, ellps,
                        radarGrid.length(), radarGrid.width(), radarGrid.wavelength(),
                        lookDirection, avg_hgt);
                checkCudaErrors(cudaPeekAtLastError());
                checkCudaErrors(cudaDeviceSynchronize());
            }

            checkCudaErrors(cudaMemcpy(&out[0], out_d, radarGrid.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost));
            out_raster.setBlock(&out[0], 0, 0, radarGrid.width(), radarGrid.length());
}
    }
}}
