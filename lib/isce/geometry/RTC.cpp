//-*- C++ -*-
//-*- coding: utf-8 -*-

#include "RTC.h"

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

#include <isce/core/Constants.h>
#include <isce/core/DateTime.h>
#include <isce/core/Ellipsoid.h>

#include <isce/geometry/geometry.h>
#include <isce/geometry/Topo.h>

using isce::core::cartesian_t;
using isce::core::Mat3;
using isce::core::Vec3;

double computeUpsamplingFactor(const isce::geometry::DEMInterpolator& dem_interp,
                               const isce::product::RadarGridParameters & radarGrid,
                               const isce::core::Ellipsoid& ellps) {
    // Create a projection object from the DEM interpolator
    isce::core::ProjectionBase * proj = isce::core::createProj(dem_interp.epsgCode());

    // Get middle XY coordinate in DEM coords, lat/lon, and ECEF XYZ
    Vec3 demXY{dem_interp.midX(), dem_interp.midY(), 0.0};
    const Vec3 xyz0 = ellps.lonLatToXyz(proj->inverse(demXY));

    // Repeat for middle coordinate + deltaX
    demXY[0] += dem_interp.deltaX();
    const Vec3 xyz1 = ellps.lonLatToXyz(proj->inverse(demXY));

    // Repeat for middle coordinate + deltaX + deltaY
    demXY[1] += dem_interp.deltaY();
    const Vec3 xyz2 = ellps.lonLatToXyz(proj->inverse(demXY));

    delete proj;

    // Estimate width/length of DEM pixel
    const double dx = (xyz1 - xyz0).norm();
    const double dy = (xyz2 - xyz1).norm();

    // Compute area of DEM pixel
    const double demArea = dx * dy;

    // Compute area of radar pixel (for now, just use spacing in range direction)
    const double radarArea = radarGrid.rangePixelSpacing() * radarGrid.rangePixelSpacing();

    // Upsampling factor is the ratio
    return std::sqrt(demArea / radarArea);
}

void isce::geometry::facetRTC(isce::product::Product& product,
                              isce::io::Raster& dem,
                              isce::io::Raster& out_raster,
                              char frequency) {

    isce::core::Orbit orbit = product.metadata().orbit();
    isce::product::RadarGridParameters radarGrid(product, frequency, 1, 1);
    int lookSide = product.lookSide();

    // Get a copy of the Doppler LUT; allow for out-of-bounds extrapolation
    isce::core::LUT2d<double> dop = product.metadata().procInfo().dopplerCentroid(frequency);
    dop.boundsError(false);

    facetRTC(radarGrid,
            orbit,
            dop,
            dem,
            out_raster,
            lookSide);

}

void isce::geometry::facetRTC(const isce::product::RadarGridParameters& radarGrid,
                            const isce::core::Orbit& orbit,
                            const isce::core::LUT2d<double>& dop,
                            isce::io::Raster& dem,
                            isce::io::Raster& out_raster,
                            const int lookSide) {

    isce::core::Ellipsoid ellps(isce::core::EarthSemiMajorAxis,
                            isce::core::EarthEccentricitySquared);

    isce::geometry::Topo topo(radarGrid, orbit, dop, ellps, lookSide);
    topo.orbitMethod(isce::core::orbitInterpMethod::HERMITE_METHOD);

    const double start = radarGrid.sensingStart();
    const double   end = radarGrid.sensingStop();
    const double pixazm = (end - start) / radarGrid.length(); // azimuth difference per pixel

    const double r0 = radarGrid.startingRange();
    const double dr = radarGrid.rangePixelSpacing();

    // Bounds for valid RDC coordinates
    double xbound = radarGrid.width()  - 1.0;
    double ybound = radarGrid.length() - 1.0;

    // Output raster
    float* out = new float[radarGrid.length() * radarGrid.width()]();

    // ------------------------------------------------------------------------
    // Main code: decompose DEM into facets, compute RDC coordinates
    // ------------------------------------------------------------------------

    isce::geometry::DEMInterpolator dem_interp(0, isce::core::dataInterpMethod::BIQUINTIC_METHOD);

    // Determine DEM bounds
    topo.computeDEMBounds(dem, dem_interp, 0, radarGrid.length());

    // Enter loop to read in SLC range/azimuth coordinates and compute area
    std::cout << std::endl;

    const float upsample_factor = computeUpsamplingFactor(dem_interp, radarGrid, ellps);

    const size_t imax = dem_interp.length() * upsample_factor;
    const size_t jmax = dem_interp.width()  * upsample_factor;

    const size_t progress_block = imax*jmax/100;
    size_t numdone = 0;

    const isce::core::ProjectionBase* proj = isce::core::createProj(dem_interp.epsgCode());

    // Loop over DEM facets
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (size_t ii = 0; ii < imax; ++ii) {
        for (size_t jj = 0; jj < jmax; ++jj) {

            #pragma omp atomic
            numdone++;

            if (numdone % progress_block == 0)
                #pragma omp critical
                printf("\rRTC progress: %d%%", (int) (numdone * 1e2 / (imax * jmax))),
                    fflush(stdout);

            // Central DEM coordinates of facets
            const double dem_ymid = dem_interp.yStart() + dem_interp.deltaY() * (ii + 0.5) / upsample_factor;
            const double dem_xmid = dem_interp.xStart() + dem_interp.deltaX() * (jj + 0.5) / upsample_factor;

            double a, r;
            const Vec3 inputDEM{dem_xmid, dem_ymid,
                dem_interp.interpolateXY(dem_xmid, dem_ymid)};
            // Compute facet-central LLH vector
            const Vec3 inputLLH = proj->inverse(inputDEM);
            //Should incorporate check on return status here
            isce::geometry::geo2rdr(inputLLH, ellps, orbit, dop,
                    a, r, radarGrid.wavelength(), 1e-4, 100, 1e-4);
            const float azpix = (a - start) / pixazm;
            const float ranpix = (r - r0) / dr;

            // Establish bounds for bilinear weighting model
            const float x1 = std::floor(ranpix);
            const float x2 = x1 + 1.0;
            const float y1 = std::floor(azpix);
            const float y2 = y1 + 1.0;

            // Check to see if pixel lies in valid RDC range
            if (ranpix < 0.0 or x2 > xbound or azpix < 0.0 or y2 > ybound)
                continue;

            // Current x/y-coords in DEM
            const double dem_y0 = dem_interp.yStart() + ii * dem_interp.deltaY() / upsample_factor;
            const double dem_y1 = dem_y0 + dem_interp.deltaY() / upsample_factor;
            const double dem_x0 = dem_interp.xStart() + dem_interp.deltaX() * jj / upsample_factor;
            const double dem_x1 = dem_x0 + dem_interp.deltaX() / upsample_factor;

            // Set DEM-coordinate corner vectors
            const Vec3 dem00 = {dem_x0, dem_y0,
                dem_interp.interpolateXY(dem_x0, dem_y0)};
            const Vec3 dem01 = {dem_x0, dem_y1,
                dem_interp.interpolateXY(dem_x0, dem_y1)};
            const Vec3 dem10 = {dem_x1, dem_y0,
                dem_interp.interpolateXY(dem_x1, dem_y0)};
            const Vec3 dem11 = {dem_x1, dem_y1,
                dem_interp.interpolateXY(dem_x1, dem_y1)};

            // Convert to XYZ
            const Vec3 xyz00 = ellps.lonLatToXyz(proj->inverse(dem00));
            const Vec3 xyz01 = ellps.lonLatToXyz(proj->inverse(dem01));
            const Vec3 xyz10 = ellps.lonLatToXyz(proj->inverse(dem10));
            const Vec3 xyz11 = ellps.lonLatToXyz(proj->inverse(dem11));

            // Compute normal vectors for each facet
            const Vec3 normalFacet1 = normalPlane(xyz00, xyz10, xyz01);
            const Vec3 normalFacet2 = normalPlane(xyz01, xyz10, xyz11);

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
            isce::core::cartesian_t xyz_plat, vel;
            orbit.interpolateWGS84Orbit(a, xyz_plat, vel);
            const Vec3 lookXYZ = (xyz_plat - xyz_mid).unitVec();

            // Compute dot product between each facet and look vector
            const double cosIncFacet1 = lookXYZ.dot(normalFacet1);
            const double cosIncFacet2 = lookXYZ.dot(normalFacet2);
            // If facets are not illuminated by radar, skip
            if (cosIncFacet1 < 0. or cosIncFacet2 < 0.) {
                continue;
            }

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
            #pragma omp atomic
            out[radarGrid.width() * iy1 + ix1] += area * Wrc * Wac;
            #pragma omp atomic
            out[radarGrid.width() * iy1 + ix2] += area * Wr * Wac;
            #pragma omp atomic
            out[radarGrid.width() * iy2 + ix1] += area * Wrc * Wa;
            #pragma omp atomic
            out[radarGrid.width() * iy2 + ix2] += area * Wr * Wa;
        }
    }

    delete proj;

    float max_hgt, avg_hgt;
    pyre::journal::info_t info("facet_calib");
    dem_interp.computeHeightStats(max_hgt, avg_hgt, info);
    isce::geometry::DEMInterpolator flat_interp(avg_hgt);

    // Compute the flat earth incidence angle correction applied by UAVSAR processing
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (size_t i = 0; i < radarGrid.length(); ++i) {
        for (size_t j = 0; j < radarGrid.width(); ++j) {

            isce::core::cartesian_t xyz_plat, vel;
            orbit.interpolateWGS84Orbit(start + i * pixazm, xyz_plat, vel);

            // Slant range for current pixel
            const double slt_range = r0 + j * dr;

            // Get LLH and XYZ coordinates for this azimuth/range
            isce::core::cartesian_t targetLLH, targetXYZ;
            targetLLH[2] = avg_hgt; // initialize first guess
            isce::geometry::rdr2geo(start + i * pixazm, slt_range, 0, orbit, ellps,
                    flat_interp, targetLLH, radarGrid.wavelength(), lookSide,
                    1e-4, 20, 20, isce::core::HERMITE_METHOD);

            // Computation of ENU coordinates around ground target
            ellps.lonLatToXyz(targetLLH, targetXYZ);
            const Vec3 satToGround = targetXYZ - xyz_plat;
            const Mat3 xyz2enu = Mat3::xyzToEnu(targetLLH[1], targetLLH[0]);
            const Vec3 enu = xyz2enu.dot(satToGround);

            // Compute incidence angle components
            const double costheta = std::abs(enu[2]) / enu.norm();
            const double sintheta = std::sqrt(1. - costheta*costheta);

            out[radarGrid.width() * i + j] *= sintheta;
        }
    }
    std::cout << std::endl;

    out_raster.setBlock(out, 0, 0, radarGrid.width(), radarGrid.length());
    delete[] out;
}
