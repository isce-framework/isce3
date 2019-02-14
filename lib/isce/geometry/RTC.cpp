//-*- C++ -*-
//-*- coding: utf-8 -*-

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
#include "isce/geometry/geometry.h"
#include "isce/geometry/RTC.h"
#include "isce/geometry/Topo.h"

// Function to compute normal vector to a plane given three points
std::array<double, 3> computePlaneNormal(std::array<double, 3> & x1,
    std::array<double, 3> & x2, std::array<double, 3>& x3) {

    std::array<double, 3> x12, x13, n, nhat;

    for (int i = 0; i < 3; i++) {
        x12[i] = x2[i] - x1[i];
        x13[i] = x3[i] - x1[i];
    }

    isce::core::LinAlg::cross(x13, x12, n);
    isce::core::LinAlg::unitVec(n, nhat);

    return nhat;
}

double computeUpsamplingFactor(const isce::geometry::DEMInterpolator& dem_interp,
                               const isce::product::ConfigParameters & param,
                               const isce::core::Ellipsoid& ellps) {
    // Create a projection object from the DEM interpolator
    isce::core::ProjectionBase * proj = isce::core::createProj(dem_interp.epsgCode());

    // Get middle XY coordinate in DEM coords, lat/lon, and ECEF XYZ
    isce::core::cartesian_t demXY{dem_interp.midX(), dem_interp.midY(), 0.0}, llh;
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
    const double radarArea = param.rangePixelSpacing() * param.rangePixelSpacing();

    // Upsampling factor is the ratio
    return std::sqrt(demArea / radarArea);
}

void isce::geometry::facetRTC(isce::product::Product& product,
                              isce::io::Raster& dem,
                              isce::io::Raster& out_raster,
                              char frequency) {
    using isce::core::LinAlg;
    const double RAD = M_PI / 180.;

    isce::core::Ellipsoid ellps(isce::core::EarthSemiMajorAxis,
                                isce::core::EarthEccentricitySquared);
    isce::core::Orbit orbit = product.metadata().orbit();
    isce::core::LUT2d<double> dop = product.metadata().procInfo().dopplerCentroid(frequency);
    isce::product::ConfigParameters param(product, frequency);
    isce::geometry::Topo topo(product);
    topo.orbitMethod(isce::core::orbitInterpMethod::HERMITE_METHOD);
    int lookSide = product.lookSide();

    const double start = param.sensingStart();
    const double   end = param.sensingStop();
    const double pixazm = (end - start) / param.length(); // azimuth difference per pixel

    const double r0 = param.startingRange();
    const double dr = param.rangePixelSpacing();

    // Initialize other ISCE objects
    isce::core::Peg peg;
    isce::core::Pegtrans ptm;
    ptm.radarToXYZ(ellps, peg);

    // Bounds for valid RDC coordinates
    double xbound = param.width()  - 1.0;
    double ybound = param.length() - 1.0;

    // Output raster
    float* out = new float[param.length() * param.width()]();

    // ------------------------------------------------------------------------
    // Main code: decompose DEM into facets, compute RDC coordinates
    // ------------------------------------------------------------------------

    isce::geometry::DEMInterpolator dem_interp(0, isce::core::dataInterpMethod::BIQUINTIC_METHOD);

    // Determine DEM bounds
    topo.computeDEMBounds(dem, dem_interp, 0, param.length());

    // Enter loop to read in SLC range/azimuth coordinates and compute area
    std::cout << std::endl;

    const float upsample_factor = computeUpsamplingFactor(dem_interp, param, ellps);

    const size_t imax = dem_interp.length() * upsample_factor;
    const size_t jmax = dem_interp.width()  * upsample_factor;

    const size_t progress_block = imax*jmax/100;
    size_t numdone = 0;

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

            isce::core::cartesian_t llh00, llh01, llh10, llh11,
                                    xyz00, xyz01, xyz10, xyz11, xyz_mid,
                                    P00_01, P00_10, P10_01, P11_01, P11_10,
                                    lookXYZ, normalFacet1, normalFacet2;

            // Central latitude/longitude of facets
            const double lat_mid = dem_interp.yStart() + dem_interp.deltaY() * (ii + 0.5) / upsample_factor;
            const double lon_mid = dem_interp.xStart() + dem_interp.deltaX() * (jj + 0.5) / upsample_factor;

            double a, r;
            isce::core::cartesian_t inputLLH{lon_mid*RAD, lat_mid*RAD,
                dem_interp.interpolateXY(lon_mid, lat_mid)};
            isce::geometry::geo2rdr(inputLLH, ellps, orbit, dop,
                    a, r, param.wavelength(), 1e-4, 100, 1e-4);
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

            // Corner latitude/longitude
            const double lat0 = dem_interp.yStart() + ii * dem_interp.deltaY() / upsample_factor;
            const double lat1 = lat0 + dem_interp.deltaY() / upsample_factor;
            const double lon0 = dem_interp.xStart() + dem_interp.deltaX() * jj / upsample_factor;
            const double lon1 = lon0 + dem_interp.deltaX() / upsample_factor;

            // Set LLH vectors
            llh00 = {RAD*lon0, RAD*lat0,
                dem_interp.interpolateXY(lon0, lat0)};
            llh01 = {RAD*lon0, RAD*lat1,
                dem_interp.interpolateXY(lon0, lat1)};
            llh10 = {RAD*lon1, RAD*lat0,
                dem_interp.interpolateXY(lon1, lat0)};
            llh11 = {RAD*lon1, RAD*lat1,
                dem_interp.interpolateXY(lon1, lat1)};

            // Convert to XYZ
            ellps.lonLatToXyz(llh00, xyz00);
            ellps.lonLatToXyz(llh01, xyz01);
            ellps.lonLatToXyz(llh10, xyz10);
            ellps.lonLatToXyz(llh11, xyz11);

            // Compute normal vectors for each facet
            normalFacet1 = computePlaneNormal(xyz00, xyz10, xyz01);
            normalFacet2 = computePlaneNormal(xyz01, xyz10, xyz11);

            // Compute vectors associated with facet sides
            LinAlg::linComb(1.0, xyz00, -1.0, xyz01, P00_01);
            LinAlg::linComb(1.0, xyz00, -1.0, xyz10, P00_10);
            LinAlg::linComb(1.0, xyz10, -1.0, xyz01, P10_01);
            LinAlg::linComb(1.0, xyz11, -1.0, xyz01, P11_01);
            LinAlg::linComb(1.0, xyz11, -1.0, xyz10, P11_10);

            // Side lengths
            const double p00_01 = LinAlg::norm(P00_01);
            const double p00_10 = LinAlg::norm(P00_10);
            const double p10_01 = LinAlg::norm(P10_01);
            const double p11_01 = LinAlg::norm(P11_01);
            const double p11_10 = LinAlg::norm(P11_10);

            // Semi-perimeters
            const float h1 = 0.5 * (p00_01 + p00_10 + p10_01);
            const float h2 = 0.5 * (p11_01 + p11_10 + p10_01);

            // Heron's formula to get area of facets in XYZ coordinates
            const float AP1 = std::sqrt(h1 * (h1 - p00_01) * (h1 - p00_10) * (h1 - p10_01));
            const float AP2 = std::sqrt(h2 * (h2 - p11_01) * (h2 - p11_10) * (h2 - p10_01));

            // Compute look angle from sensor to ground
            ellps.lonLatToXyz(inputLLH, xyz_mid);
            isce::core::cartesian_t xyz_plat, vel;
            orbit.interpolateWGS84Orbit(a, xyz_plat, vel);
            lookXYZ = {xyz_plat[0] - xyz_mid[0],
                       xyz_plat[1] - xyz_mid[1],
                       xyz_plat[2] - xyz_mid[2]};
            double norm = LinAlg::norm(lookXYZ);
            lookXYZ[0] /= norm;
            lookXYZ[1] /= norm;
            lookXYZ[2] /= norm;

            // Compute dot product between each facet and look vector
            const double cosIncFacet1 = LinAlg::dot(lookXYZ, normalFacet1);
            const double cosIncFacet2 = LinAlg::dot(lookXYZ, normalFacet2);
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
            out[param.width() * iy1 + ix1] += area * Wrc * Wac;
            #pragma omp atomic
            out[param.width() * iy1 + ix2] += area * Wr * Wac;
            #pragma omp atomic
            out[param.width() * iy2 + ix1] += area * Wrc * Wa;
            #pragma omp atomic
            out[param.width() * iy2 + ix2] += area * Wr * Wa;
        }
    }

    float max_hgt, avg_hgt;
    pyre::journal::info_t info("facet_calib");
    dem_interp.computeHeightStats(max_hgt, avg_hgt, info);
    isce::geometry::DEMInterpolator flat_interp(avg_hgt);

    // Compute the flat earth incidence angle correction applied by UAVSAR processing
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (size_t i = 0; i < param.length(); ++i) {
        for (size_t j = 0; j < param.width(); ++j) {

            isce::core::cartesian_t xyz_plat, vel;
            orbit.interpolateWGS84Orbit(start + i * pixazm, xyz_plat, vel);

            // Slant range for current pixel
            const double slt_range = r0 + j * dr;

            // Get LLH and XYZ coordinates for this azimuth/range
            isce::core::cartesian_t targetLLH, targetXYZ;
            targetLLH[2] = avg_hgt; // initialize first guess
            isce::geometry::rdr2geo(start + i * pixazm, slt_range, 0, orbit, ellps,
                    flat_interp, targetLLH, param.wavelength(), lookSide,
                    1e-4, 20, 20, isce::core::HERMITE_METHOD);

            // Computation of ENU coordinates around ground target
            isce::core::cartesian_t satToGround, enu;
            isce::core::cartmat_t enumat, xyz2enu;
            ellps.lonLatToXyz(targetLLH, targetXYZ);
            LinAlg::linComb(1.0, targetXYZ, -1.0, xyz_plat, satToGround);
            LinAlg::enuBasis(targetLLH[1], targetLLH[0], enumat);
            LinAlg::tranMat(enumat, xyz2enu);
            LinAlg::matVec(xyz2enu, satToGround, enu);

            // Compute incidence angle components
            const double costheta = std::abs(enu[2]) / LinAlg::norm(enu);
            const double sintheta = std::sqrt(1. - costheta*costheta);

            out[param.width() * i + j] *= sintheta;
        }
    }
    std::cout << std::endl;

    out_raster.setBlock(out, 0, 0, param.width(), param.length());
    delete[] out;
}
