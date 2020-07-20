// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018
//

#include "geometry.h"

#include <cmath>
#include <cstdio>
#include <limits>

#include <isce3/core/Basis.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Peg.h>
#include <isce3/core/Pixel.h>
#include <isce3/core/Poly2d.h>
#include <isce3/core/Projections.h>
#include <isce3/core/Vector.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/core/LookSide.h>
#include <isce3/product/RadarGridParameters.h>
#include <pyre/journal.h>

#include "detail/Geo2Rdr.h"
#include "detail/Rdr2Geo.h"

// pull in useful isce::core namespace
using namespace isce::core;
using isce::error::ErrorCode;
using isce::product::RadarGridParameters;

int isce::geometry::
rdr2geo(double aztime, double slantRange, double doppler, const Orbit & orbit,
        const Ellipsoid & ellipsoid, const DEMInterpolator & demInterp, Vec3 & targetLLH,
        double wvl, LookSide side, double threshold, int maxIter, int extraIter)
{
    double h0 = targetLLH[2];
    detail::Rdr2GeoParams params = {threshold, maxIter, extraIter};
    auto status =
            detail::rdr2geo(&targetLLH, aztime, slantRange, doppler, orbit,
                            demInterp, ellipsoid, wvl, side, h0, params);
    return (status == ErrorCode::Success);
}

int isce::geometry::
rdr2geo(const Pixel & pixel, const Basis & TCNbasis, const Vec3& pos, const Vec3& vel,
        const Ellipsoid & ellipsoid, const DEMInterpolator & demInterp,
        Vec3 & targetLLH, LookSide side, double threshold, int maxIter, int extraIter)
{
    double h0 = targetLLH[2];
    detail::Rdr2GeoParams params = {threshold, maxIter, extraIter};
    auto status = detail::rdr2geo(&targetLLH, pixel, TCNbasis, pos, vel,
                                  demInterp, ellipsoid, side, h0, params);
    return (status == ErrorCode::Success);
}


int isce::geometry::
rdr2geo(const Vec3& radarXYZ, const Vec3& axis, double angle,
        double range, const DEMInterpolator& dem, Vec3& targetXYZ,
        LookSide side, double threshold, int maxIter, int extraIter)
{
    if (range <= 0.0)
        return 0;
    int epsg = dem.epsgCode();
    Ellipsoid ell = makeProjection(epsg)->ellipsoid();
    // Generate TCN basis using the given axis as the velocity.
    Basis tcn(radarXYZ, axis);
    // Construct "doppler factor" with desired angle.
    Pixel pix{range, range * sin(angle), 0};
    Vec3 llh{0,0,0}; // XXX Initialize height guess of 0 m.
    int converged = isce::geometry::rdr2geo(pix, tcn, radarXYZ, axis, ell, dem,
                                    llh, side, threshold, maxIter, extraIter);
    if (converged)
        ell.lonLatToXyz(llh, targetXYZ);
    return converged;
}


template <class T>
double isce::geometry::
    _compute_doppler_aztime_diff(Vec3 dr, Vec3 satvel,
                                 T &doppler, double wavelength,
                                 double aztime, double slantRange,
                                 double deltaRange) {

    // Compute doppler
    const double dopfact = dr.dot(satvel);
    const double fdop = doppler.eval(aztime, slantRange) * 0.5 * wavelength;
    // Use forward difference to compute doppler derivative
    const double fdopder = (doppler.eval(aztime, slantRange + deltaRange) * 0.5 * wavelength - fdop)
                         / deltaRange;

    // Evaluate cost function and its derivative
    const double fn = dopfact - fdop * slantRange;
    const double c1 = -satvel.dot(satvel);
    const double c2 = (fdop / slantRange) + fdopder;
    const double fnprime = c1 + c2 * dopfact;

    const double aztime_diff = fn / fnprime;

    return aztime_diff;
}

namespace isce::geometry {
namespace {
int
_update_aztime(const Orbit & orbit,
               Vec3 satpos, Vec3 satvel, Vec3 inputXYZ,
               LookSide side, double & aztime, double & slantRange,
               double rangeMin = std::numeric_limits<double>::quiet_NaN(),
               double rangeMax = std::numeric_limits<double>::quiet_NaN())
{

    Vec3 dr;

    // Compute azimuth time spacing for coarse grid search
    const int NUM_AZTIME_TEST = 15;
    const double tstart = orbit.startTime();
    const double tend = orbit.endTime();

    int error = 1;

    // If aztime is valid (user-defined) exit
    if (aztime >= tstart && aztime <= tend)
        return !error;

    const double delta_t = (tend - tstart) / (1.0 * (NUM_AZTIME_TEST - 1));

    // Find azimuth time with minimum valid range distance to target
    double slantRange_closest = 1.0e16;
    double aztime_closest = -1000.0;
    for (int k = 0; k < NUM_AZTIME_TEST; ++k) {
        // Interpolate orbit
        aztime = tstart + k * delta_t;
        if (aztime < orbit.startTime() || aztime > orbit.endTime())
            continue;
        orbit.interpolate(&satpos, &satvel, aztime,
                          OrbitInterpBorderMode::FillNaN);
        // Compute slant range
        dr = inputXYZ - satpos;

        // Check look side (only first time)
        if (k == 0) {
            // (Left && positive) || (Right && negative)
            if ((side == LookSide::Right) ^ (dr.cross(satvel).dot(satpos) > 0)) {
                return error; // wrong look side
            }
        }

        slantRange = dr.norm();
        // Check validity
        if (!std::isnan(rangeMin) && slantRange < rangeMin)
            continue;
        if (!std::isnan(rangeMax) && slantRange > rangeMax)
            continue;

        // Update best guess
        if (slantRange < slantRange_closest) {
            slantRange_closest = slantRange;
            aztime_closest = aztime;
        }
    }

    // If we did not find a good guess, use tmid as intial guess
    if (aztime_closest < 0.0)
        aztime = orbit.midTime();
    else
        aztime = aztime_closest;
    return !error;
}
} // anonymous namespace
} // isce::geometry


int isce::geometry::
geo2rdr(const Vec3 & inputLLH, const Ellipsoid & ellipsoid, const Orbit & orbit,
        const Poly2d & doppler, double & aztime, double & slantRange,
        double wavelength, double startingRange, double rangePixelSpacing, size_t rwidth,
        LookSide side, double threshold, int maxIter, double deltaRange)
{

    Vec3 satpos, satvel, inputXYZ, dr;

    // Convert LLH to XYZ
    ellipsoid.lonLatToXyz(inputLLH, inputXYZ);

    // Pre-compute scale factor for doppler
    // const double dopscale = 0.5 * wavelength;

    // Compute minimum and maximum valid range
    const double rangeMin = startingRange;
    const double rangeMax = rangeMin + rangePixelSpacing * (rwidth - 1);

    int converged = 1;
    int error = _update_aztime(orbit, satpos, satvel, inputXYZ, side,
                               aztime, slantRange, rangeMin, rangeMax);
    if (error)
        return !converged;

    double slantRange_old = slantRange;
    // Begin iterations
    for (int i = 0; i < maxIter; ++i) {

        // Interpolate the orbit to current estimate of azimuth time
        orbit.interpolate(&satpos, &satvel, aztime, OrbitInterpBorderMode::FillNaN);

        // Compute slant range from satellite to ground point
        dr = inputXYZ - satpos;

        // Check look side (only first time)
        if (i == 0) {
            // (Left && positive) || (Right && negative)
            if ((side == LookSide::Right) ^ (dr.cross(satvel).dot(satpos) > 0)) {
                return !converged; // wrong look side
            }
        }

        slantRange = dr.norm();
        // Check convergence
        if (std::abs(slantRange - slantRange_old) < threshold)
            return converged;
        else
            slantRange_old = slantRange;

        // Update guess for azimuth time
        double aztime_diff = _compute_doppler_aztime_diff(dr, satvel,
                                                          doppler, wavelength,
                                                          aztime,
                                                          slantRange, deltaRange);

        aztime -= aztime_diff;
    }
    // If we reach this point, no convergence for specified threshold
    return !converged;
}

int isce::geometry::
geo2rdr(const Vec3 & inputLLH, const Ellipsoid & ellipsoid, const Orbit & orbit,
        const LUT2d<double> & doppler, double & aztime, double & slantRange,
        double wavelength, LookSide side, double threshold, int maxIter,
        double deltaRange)
{
    double t0 = aztime;
    detail::Geo2RdrParams params = {threshold, maxIter, deltaRange};
    auto status =
            detail::geo2rdr(&aztime, &slantRange, inputLLH, ellipsoid, orbit,
                            doppler, wavelength, side, t0, params);
    return (status == ErrorCode::Success);
}

// Utility function to compute geographic bounds for a radar grid
void isce::geometry::
computeDEMBounds(const Orbit & orbit,
                 const Ellipsoid & ellipsoid,
                 const LUT2d<double> & doppler,
                 const RadarGridParameters & radarGrid,
                 size_t xoff,
                 size_t yoff,
                 size_t xsize,
                 size_t ysize,
                 double margin,
                 double & min_lon,
                 double & min_lat,
                 double & max_lon,
                 double & max_lat)
{
    // Initialize geographic bounds
    min_lon = 1.0e64;
    max_lon = -1.0e64;
    min_lat = 1.0e64;
    max_lat = -1.0e64;

    // Initialize journal
    pyre::journal::warning_t warning("isce.geometry.extractDEM");

    isce::core::LookSide lookSide = radarGrid.lookSide();

    // Skip factors along azimuth and range
    const int askip = std::max((int) ysize / 10, 1);
    const int rskip = xsize / 10;

    // Construct vectors of range/azimuth indices traversing the perimeter of the radar frame

    // Top edge
    std::vector<int> azInd, rgInd;
    for (int j = 0; j < xsize; j += rskip) {
        azInd.push_back(yoff);
        rgInd.push_back(j + xoff);
    }

    // Right edge
    for (int i = 0; i < ysize; i += askip) {
        azInd.push_back(i + yoff);
        rgInd.push_back(xsize + xoff);
    }

    // Bottom edge
    for (int j = xsize; j > 0; j -= rskip) {
        azInd.push_back(yoff + ysize - 1);
        rgInd.push_back(j + xoff);
    }

    // Left edge
    for (int i = ysize; i > 0; i -= askip) {
        azInd.push_back(i + yoff);
        rgInd.push_back(xoff);
    }

    // Loop over the indices
    for (size_t i = 0; i < rgInd.size(); ++i) {

        // Compute satellite azimuth time
        const double tline = radarGrid.sensingTime(azInd[i]);

        // Get state vector
        Vec3 xyzsat, velsat;
        orbit.interpolate(&xyzsat, &velsat, tline,
                          OrbitInterpBorderMode::FillNaN);
        // Save state vector
        const Vec3 pos = xyzsat;
        const Vec3 vel = velsat;

        // Get geocentric TCN basis using satellite basis
        Basis TCNbasis(pos, vel);

        // Compute satellite velocity and height
        Vec3 satLLH;
        const double satVmag = velsat.norm();
        ellipsoid.xyzToLonLat(xyzsat, satLLH);

        // Get proper slant range and Doppler factor
        const size_t rbin = rgInd[i];
        const double rng = radarGrid.slantRange(rbin);
        const double dopfact = (0.5 * radarGrid.wavelength() * (doppler.eval(tline, rng) /
                                satVmag)) * rng;

        // Store in Pixel object
        Pixel pixel(rng, dopfact, rbin);

        // Run topo for one iteration for two different heights
        Vec3 llh {0., 0., 0.};
        std::array<double, 2> testHeights = {-500.0, 1000.0};
        for (int k = 0; k < 2; ++k) {

            // If slant range vector doesn't hit ground, pick nadir point
            if (rng <= (satLLH[2] - testHeights[k] + 1.0)) {
                for (int idx = 0; idx < 3; ++idx) {
                    llh[idx] = satLLH[idx];
                }
                warning << "Possible near nadir imaging" << pyre::journal::endl;
            } else {
                // Create dummy DEM interpolator with constant height
                DEMInterpolator constDEM(testHeights[k]);
                // Run radar->geo for 1 iteration
                rdr2geo(pixel, TCNbasis, pos, vel, ellipsoid, constDEM, llh,
                        lookSide, 1.0e-5, 1, 0);
            }

            // Update bounds
            min_lat = std::min(min_lat, llh[1]);
            max_lat = std::max(max_lat, llh[1]);
            min_lon = std::min(min_lon, llh[0]);
            max_lon = std::max(max_lon, llh[0]);
        }
    }

    // Account for margins
    min_lon -= margin;
    max_lon += margin;
    min_lat -= margin;
    max_lat += margin;

}

// end of file
