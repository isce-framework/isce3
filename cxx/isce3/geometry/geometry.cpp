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
#include <utility>

#include <pyre/journal.h>

#include <isce3/core/Basis.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/LookSide.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Peg.h>
#include <isce3/core/Pixel.h>
#include <isce3/core/Poly2d.h>
#include <isce3/core/Projections.h>
#include <isce3/core/Vector.h>
#include <isce3/except/Error.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/product/RadarGridParameters.h>

#include "detail/Geo2Rdr.h"
#include "detail/Rdr2Geo.h"

// pull in useful isce3::core namespace
using namespace isce3::core;
using isce3::error::ErrorCode;
using isce3::product::RadarGridParameters;

int isce3::geometry::rdr2geo(double aztime, double slantRange, double doppler,
        const Orbit& orbit, const Ellipsoid& ellipsoid,
        const DEMInterpolator& demInterp, Vec3& targetLLH, double wvl,
        LookSide side, double threshold, int maxIter, int extraIter)
{
    double h0 = targetLLH[2];
    detail::Rdr2GeoParams params = {threshold, maxIter, extraIter};
    auto status = detail::rdr2geo(&targetLLH, aztime, slantRange, doppler,
            orbit, demInterp, ellipsoid, wvl, side, h0, params);
    return (status == ErrorCode::Success);
}

int isce3::geometry::rdr2geo(const Pixel& pixel, const Basis& TCNbasis,
        const Vec3& pos, const Vec3& vel, const Ellipsoid& ellipsoid,
        const DEMInterpolator& demInterp, Vec3& targetLLH, LookSide side,
        double threshold, int maxIter, int extraIter)
{
    double h0 = targetLLH[2];
    detail::Rdr2GeoParams params = {threshold, maxIter, extraIter};
    auto status = detail::rdr2geo(&targetLLH, pixel, TCNbasis, pos, vel,
            demInterp, ellipsoid, side, h0, params);
    return (status == ErrorCode::Success);
}

int isce3::geometry::rdr2geo(const Vec3& radarXYZ, const Vec3& axis,
        double angle, double range, const DEMInterpolator& dem, Vec3& targetXYZ,
        LookSide side, double threshold, int maxIter, int extraIter)
{
    if (range <= 0.0)
        return 0;
    int epsg = dem.epsgCode();
    Ellipsoid ell = makeProjection(epsg)->ellipsoid();
    // Generate TCN basis using the given axis as the velocity.
    Basis tcn(radarXYZ, axis);
    // Construct "doppler factor" with desired angle.
    Pixel pix {range, range * sin(angle), 0};
    Vec3 llh {0, 0, 0}; // XXX Initialize height guess of 0 m.
    int converged = isce3::geometry::rdr2geo(pix, tcn, radarXYZ, axis, ell, dem,
            llh, side, threshold, maxIter, extraIter);
    if (converged)
        ell.lonLatToXyz(llh, targetXYZ);
    return converged;
}

template<class T>
double isce3::geometry::_compute_doppler_aztime_diff(Vec3 dr, Vec3 satvel,
        T& doppler, double wavelength, double aztime, double slantRange,
        double deltaRange)
{

    // Compute doppler
    const double dopfact = dr.dot(satvel);
    const double fdop = doppler.eval(aztime, slantRange) * 0.5 * wavelength;
    // Use forward difference to compute doppler derivative
    const double fdopder =
            (doppler.eval(aztime, slantRange + deltaRange) * 0.5 * wavelength -
                    fdop) /
            deltaRange;

    // Evaluate cost function and its derivative
    const double fn = dopfact - fdop * slantRange;
    const double c1 = -satvel.dot(satvel);
    const double c2 = (fdop / slantRange) + fdopder;
    const double fnprime = c1 + c2 * dopfact;

    const double aztime_diff = fn / fnprime;

    return aztime_diff;
}

namespace isce3::geometry { namespace {
int _update_aztime(const Orbit& orbit, Vec3 satpos, Vec3 satvel, Vec3 inputXYZ,
        LookSide side, double& aztime, double& slantRange,
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
        orbit.interpolate(
                &satpos, &satvel, aztime, OrbitInterpBorderMode::FillNaN);
        // Compute slant range
        dr = inputXYZ - satpos;

        // Check look side (only first time)
        if (k == 0) {
            // (Left && positive) || (Right && negative)
            if ((side == LookSide::Right) ^
                    (dr.cross(satvel).dot(satpos) > 0)) {
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
}} // namespace isce3::geometry

int isce3::geometry::geo2rdr(const Vec3& inputLLH, const Ellipsoid& ellipsoid,
        const Orbit& orbit, const Poly2d& doppler, double& aztime,
        double& slantRange, double wavelength, double startingRange,
        double rangePixelSpacing, size_t rwidth, LookSide side,
        double threshold, int maxIter, double deltaRange)
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
    int error = _update_aztime(orbit, satpos, satvel, inputXYZ, side, aztime,
            slantRange, rangeMin, rangeMax);
    if (error)
        return !converged;

    // Newton step, initialized to zero.
    double aztime_diff = 0.0;

    // Begin iterations
    for (int i = 0; i < maxIter; ++i) {
        // Apply Newton step computed in previous iteration here so that
        // (aztime, slantRange) are always consistent on return.
        aztime -= aztime_diff;

        // Interpolate the orbit to current estimate of azimuth time
        orbit.interpolate(
                &satpos, &satvel, aztime, OrbitInterpBorderMode::FillNaN);

        // Compute slant range from satellite to ground point
        dr = inputXYZ - satpos;

        // Check look side (only first time)
        if (i == 0) {
            // (Left && positive) || (Right && negative)
            if ((side == LookSide::Right) ^
                    (dr.cross(satvel).dot(satpos) > 0)) {
                return !converged; // wrong look side
            }
        }

        slantRange = dr.norm();

        // Update guess for azimuth time
        aztime_diff = _compute_doppler_aztime_diff(dr, satvel, doppler,
                wavelength, aztime, slantRange, deltaRange);

        // Check convergence
        if (std::abs(aztime_diff) < threshold) {
            return converged;
        }
    }
    // If we reach this point, no convergence for specified threshold
    return !converged;
}

int isce3::geometry::geo2rdr(const Vec3& inputLLH, const Ellipsoid& ellipsoid,
        const Orbit& orbit, const LUT2d<double>& doppler, double& aztime,
        double& slantRange, double wavelength, LookSide side, double threshold,
        int maxIter, double deltaRange)
{
    double t0 = aztime;
    detail::Geo2RdrParams params = {threshold, maxIter, deltaRange};
    auto status = detail::geo2rdr(&aztime, &slantRange, inputLLH, ellipsoid,
            orbit, doppler, wavelength, side, t0, params);
    return (status == ErrorCode::Success);
}

// Utility function to compute geographic bounds for a radar grid
void isce3::geometry::computeDEMBounds(const Orbit& orbit,
        const Ellipsoid& ellipsoid, const LUT2d<double>& doppler,
        const RadarGridParameters& radarGrid, size_t xoff, size_t yoff,
        size_t xsize, size_t ysize, double margin, double& min_lon,
        double& min_lat, double& max_lon, double& max_lat)
{
    // Initialize geographic bounds
    min_lon = 1.0e64;
    max_lon = -1.0e64;
    min_lat = 1.0e64;
    max_lat = -1.0e64;

    // Initialize journal
    pyre::journal::warning_t warning("isce.geometry.extractDEM");

    isce3::core::LookSide lookSide = radarGrid.lookSide();

    // Skip factors along azimuth and range
    const int askip = std::max((int) ysize / 10, 1);
    const int rskip = xsize / 10;

    // Construct vectors of range/azimuth indices traversing the perimeter of
    // the radar frame

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
        orbit.interpolate(
                &xyzsat, &velsat, tline, OrbitInterpBorderMode::FillNaN);
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
        const double dopfact = (0.5 * radarGrid.wavelength() *
                                       (doppler.eval(tline, rng) / satVmag)) *
                               rng;

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

Vec3 isce3::geometry::nedVector(double lon, double lat, const Vec3& vel)
{
    const double coslat {std::cos(lat)};
    const double sinlat {std::sin(lat)};
    const double coslon {std::cos(lon)};
    const double sinlon {std::sin(lon)};
    Vec3 vned;
    vned(0) = -sinlat * coslon * vel(0) - sinlat * sinlon * vel(1) +
              coslat * vel(2);
    vned(1) = -sinlon * vel(0) + coslon * vel(1);
    vned(2) = -coslat * coslon * vel(0) - coslat * sinlon * vel(1) -
              sinlat * vel(2);
    return vned;
}

Vec3 isce3::geometry::nwuVector(double lon, double lat, const Vec3& vel)
{
    auto nwu {nedVector(lon, lat, vel)};
    nwu.tail(2) *= -1.0;
    return nwu;
}

Vec3 isce3::geometry::enuVector(double lon, double lat, const Vec3& vel)
{
    auto enu {nedVector(lon, lat, vel)};
    enu(2) *= -1.0;
    std::swap(enu(0), enu(1));
    return enu;
}

double isce3::geometry::heading(double lon, double lat, const Vec3& vel)
{
    const double coslat {std::cos(lat)};
    const double sinlat {std::sin(lat)};
    const double coslon {std::cos(lon)};
    const double sinlon {std::sin(lon)};
    return std::atan2(-sinlon * vel(0) + coslon * vel(1),
            -sinlat * coslon * vel(0) - sinlat * sinlon * vel(1) +
                    coslat * vel(2));
}

/**
 * @internal
 * A helper function to get "D" value in NED vector from ECEF vector
 * at a certain geodetic location.
 * @param[in] lon : geodetic longitude in radians
 * @param[in] lat : geodetic latitude in radians
 * @param[in] vec : 3-D Vec3 in ECEF coordinate
 * @return a double scalar representing the Down value in NED vector
 */
static double _downVal(double lon, double lat, const Vec3& vec)
{
    auto uvec {vec.normalized()};
    const double coslat {std::cos(lat)};
    const double sinlat {std::sin(lat)};
    const double coslon {std::cos(lon)};
    const double sinlon {std::sin(lon)};
    return (-coslat * coslon * uvec(0) - coslat * sinlon * uvec(1) -
            sinlat * uvec(2));
}

double isce3::geometry::slantRangeFromLookVec(
        const Vec3& pos, const Vec3& lkvec, const Ellipsoid& ellips)
{
    if (lkvec.isZero())
        throw isce3::except::InvalidArgument(
                ISCE_SRCINFO(), "Input lookvector must be non-zero vector!");
    const auto b1 {ellips.b()};
    const auto a2 {ellips.a() * ellips.a()};
    const auto b2 {b1 * b1};
    auto nlkvec {lkvec.normalized()};
    double tmpa =
            nlkvec.head(2).squaredNorm() / a2 + nlkvec(2) * nlkvec(2) / b2;
    double tmpb =
            nlkvec.head(2).dot(pos.head(2)) / a2 + nlkvec(2) * pos(2) / b2;
    double tmpc = pos.head(2).squaredNorm() / a2 + pos(2) * pos(2) / b2 - 1.0;
    double tmpx = tmpb * tmpb - tmpa * tmpc;
    if (tmpx < 0.0)
        throw isce3::except::RuntimeError(
                ISCE_SRCINFO(), "Bad inputs results in negative square root!");
    double sr = -(tmpb + std::sqrt(tmpx)) / tmpa;
    if (!(sr > 0.0))
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad inputs results in non-positive slant range!");
    return sr;
}

std::pair<int, double> isce3::geometry::srPosFromLookVecDem(double& sr,
        Vec3& tg_pos, Vec3& llh, const Vec3& sc_pos, const Vec3& lkvec,
        const DEMInterpolator& dem_interp, double hgt_err, int num_iter,
        const Ellipsoid& ellips, std::optional<double> initial_height)
{
    if (hgt_err <= 0.0 || num_iter <= 0)
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(),
                "Height Error and number of iteration "
                "must be non-zero positive values!");

    // either use provided initial height or compute the mean DEM
    // from DEM stats if not already computed!
    double mean_hgt {0};
    if (initial_height)
        mean_hgt = *initial_height;
    else
        mean_hgt = compute_mean_dem(dem_interp);
    // form a new ellipsoid whose radii adjusted by mean height
    const auto a_new = ellips.a() + mean_hgt;
    const auto b_new = ellips.b() + mean_hgt;
    const auto e2_new = 1.0 - (b_new * b_new) / (a_new * a_new);
    // initial guess of slant range per new ellipsoid
    sr = slantRangeFromLookVec(sc_pos, lkvec, Ellipsoid(a_new, e2_new));
    int cnt {0};
    double abs_hgt_dif;
    do {
        tg_pos = sr * lkvec + sc_pos;
        llh = ellips.xyzToLonLat(tg_pos);
        auto dem_hgt = dem_interp.interpolateLonLat(llh(0), llh(1));
        auto hgt_dif = dem_hgt - llh(2);
        sr += hgt_dif / _downVal(llh(0), llh(1), tg_pos);
        abs_hgt_dif = std::abs(hgt_dif);
        ++cnt;
    } while (cnt < num_iter && abs_hgt_dif > hgt_err);
    if (cnt == num_iter && abs_hgt_dif > hgt_err)
        std::cerr << "Warning: reached max iterations " << cnt
                  << " with height error " << abs_hgt_dif << " (m)!\n";
    return std::make_pair(cnt, abs_hgt_dif);
}

/**
 * @internal
 * A helper function to get two geometry values "platform-height +
 * alongtrack-range-curvature" and "dem-height + alongtrack-range-curvature"
 * from orbit and mean DEM at a certain azimuth time for a certain ellipsoidal
 * planet. This function will be used in "lookIncAngFromSlantRange" functions.
 * @param[in] orbit: isce3 orbit object.
 * @param[in] az_time (optional): relative azimuth time in seconds w.r.t
 * reference epoch time of orbit object. If not speficied or set to {} or
 * std::nullopt, the mid time of orbit will be used as azimuth time.
 * @param[in] dem_interp: DEMInterpolator object wrt reference ellipsoid.
 * @param[in] ellips: Ellipsoid object.
 * @return a tuple of two scalars, "platform-height +
 * alongtrack-range-curvature" and "dem-height + alongtrack-range-curvature"
 * @exception RuntimeError
 */
static std::tuple<double, double> _get_rgcurv_plus_hgt(
        const isce3::core::Orbit& orbit, std::optional<double> az_time,
        const isce3::geometry::DEMInterpolator& dem_interp,
        const isce3::core::Ellipsoid& ellips)
{
    // set azimuth time to midtime of orbit if not specified
    if (!az_time)
        *az_time = orbit.midTime();
    // get the pos/vel of S/C at azimuth time
    isce3::core::Vec3 sc_pos, sc_vel;
    auto err_code = orbit.interpolate(&sc_pos, &sc_vel, *az_time);
    if (err_code != isce3::error::ErrorCode::Success)
        throw isce3::except::RuntimeError(
                ISCE_SRCINFO(), isce3::error::getErrorString(err_code));
    // get LLH of S/C
    auto sc_llh = ellips.xyzToLonLat(sc_pos);
    // get heading of S/C
    auto sc_hdg = isce3::geometry::heading(sc_llh(0), sc_llh(1), sc_vel);
    // get ellipsoid range curvature along flight track / heading
    auto rg_curv = ellips.rDir(sc_hdg, sc_llh(1));
    // get mean dem height
    auto dem_hgt = compute_mean_dem(dem_interp);
    return {sc_llh(2) + rg_curv, dem_hgt + rg_curv};
}

std::tuple<double, double> isce3::geometry::lookIncAngFromSlantRange(
        double slant_range, const isce3::core::Orbit& orbit,
        std::optional<double> az_time, const DEMInterpolator& dem_interp,
        const isce3::core::Ellipsoid& ellips)
{
    // get combinations of range curvature with platform height as well as mean
    // DEM height
    double schgt_plus_rgcurv;
    double demhgt_plus_rgcurv;
    std::tie(schgt_plus_rgcurv, demhgt_plus_rgcurv) =
            _get_rgcurv_plus_hgt(orbit, az_time, dem_interp, ellips);

    // calculate look angle
    double lk_ang = std::acos(
            (slant_range * slant_range + schgt_plus_rgcurv * schgt_plus_rgcurv -
                    demhgt_plus_rgcurv * demhgt_plus_rgcurv) /
            (2.0 * slant_range * schgt_plus_rgcurv));
    // check if look angle is a valid value
    const auto half_pi = M_PI / 2.0;
    if (std::isnan(lk_ang) || !(lk_ang > 0.0 && lk_ang < half_pi))
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad input values result in nan or unacceptable look angle!");

    // calculate incidence angle
    double inc_ang = lk_ang + std::asin(std::sin(lk_ang) * slant_range /
                                        demhgt_plus_rgcurv);
    return {lk_ang, inc_ang};
}

std::tuple<Eigen::ArrayXd, Eigen::ArrayXd>
isce3::geometry::lookIncAngFromSlantRange(
        const Eigen::Ref<const Eigen::ArrayXd>& slant_range,
        const isce3::core::Orbit& orbit, std::optional<double> az_time,
        const DEMInterpolator& dem_interp, const isce3::core::Ellipsoid& ellips)
{
    // get combinations of along-track range curvature with platform height as
    // well as with mean DEM height
    double schgt_plus_rgcurv;
    double demhgt_plus_rgcurv;
    std::tie(schgt_plus_rgcurv, demhgt_plus_rgcurv) =
            _get_rgcurv_plus_hgt(orbit, az_time, dem_interp, ellips);

    // define a lambda function for look angle and slant range calculation
    auto est_look = [=](double sr) {
        return std::acos((sr * sr + schgt_plus_rgcurv * schgt_plus_rgcurv -
                                 demhgt_plus_rgcurv * demhgt_plus_rgcurv) /
                         (2.0 * sr * schgt_plus_rgcurv));
    };

    auto est_incidence = [=](double sr, double lka) {
        return lka + std::asin(std::sin(lka) * sr / demhgt_plus_rgcurv);
    };

    // Simply calculate look angles for (min,max) slant range vector
    // Note that no need to assume to  have uniform-spacing and/or monotonic
    // slant range values! simply check (min, max) look angles to make sure they
    // are within reasonable range
    const auto half_pi = M_PI / 2.0;
    auto lka_min = est_look(slant_range.minCoeff());
    if (std::isnan(lka_min) || !(lka_min > 0.0 && lka_min < half_pi))
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad input values result in nan or "
                "unacceptable min look angle!");
    auto lka_max = est_look(slant_range.maxCoeff());
    if (std::isnan(lka_max) || !(lka_max > lka_min && lka_max < half_pi))
        throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                "Bad input values result in nan or "
                "unacceptable max look angle!");

    // allocate output vectors with proper size
    const auto len = slant_range.size();
    Eigen::ArrayXd lka_all(len);
    Eigen::ArrayXd inca_all(len);
    // loop over all slant ranges and calculate both look angle and incidecne
    // angle
    for (Eigen::Index idx = 0; idx < len; ++idx) {
        lka_all(idx) = est_look(slant_range(idx));
        inca_all(idx) = est_incidence(slant_range(idx), lka_all(idx));
    }
    return {lka_all, inca_all};
}

double isce3::geometry::compute_mean_dem(const DEMInterpolator& dem)
{
    if (dem.haveRaster()) {
        if (dem.haveStats())
            return dem.meanHeight();
        else {
            double mean = 0.0;
            auto n_valid = dem.length() * dem.width();
            const auto d = dem.data();
// loop over all values in DEM raster
#pragma omp parallel for reduction(+ : mean) reduction(- : n_valid)
            for (size_t idx = 0; idx < dem.length() * dem.width(); ++idx) {
                auto val = d[idx];
                if (std::isnan(val)) {
                    n_valid--;
                    continue;
                }
                mean += val;
            }
            if (n_valid != 0)
                mean /= n_valid;
            return mean;
        }
    }
    return dem.refHeight();
}
// end of file
