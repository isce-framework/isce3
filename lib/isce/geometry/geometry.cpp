// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018
//

#include <cmath>
#include <cstdio>

// isce::core
#include <isce/core/LinAlg.h>
#include <isce/core/Peg.h>

// isce::geometry
#include "geometry.h"

// pull in useful isce::core namespace
using isce::core::Basis;
using isce::core::LinAlg;
using isce::core::Ellipsoid;
using isce::core::Metadata;
using isce::core::Orbit;
using isce::core::Pegtrans;
using isce::core::Pixel;
using isce::core::Poly2d;
using isce::core::StateVector;
using isce::product::ImageMode;

/** @param[in] aztime azimuth time corresponding to line of interest
 * @param[in] slantRange slant range corresponding to pixel of interest
 * @param[in] doppler doppler model value corresponding to line,pixel
 * @param[in] orbit Orbit object
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] demInterp DEMInterpolator object
 * @param[out] targetLLH output Lon/Lat/Hae corresponding to aztime and slantRange
 * @param[in] wvl imaging wavelength
 * @param[in] side +1 for left and -1 for right
 * @param[in] threshold Distance threshold for convergence
 * @param[in] maxIter Number of primary iterations
 * @param[in] extraIter Number of secondary iterations
 * @param[in] orbitMethod Orbit interpolation method
 *
 * This is meant to be the light version of isce::geometry::Topo and not meant to be used for processing large number of targets of interest. Note that doppler and wavelength are meant for completeness and this method can be used with both Native and Zero Doppler geometries. For details of the algorithm, see the \ref overview_geometry "geometry overview".*/ 
int isce::geometry::
rdr2geo(double aztime, double slantRange, double doppler, const Orbit & orbit,
        const Ellipsoid & ellipsoid, const DEMInterpolator & demInterp, cartesian_t & targetLLH,
        double wvl, int side, double threshold, int maxIter, int extraIter,
        isce::core::orbitInterpMethod orbitMethod) {
    /*
    Interpolate Orbit to azimuth time, compute TCN basis, and estimate geographic
    coordinates.
    */

    // Interpolate orbit to get state vector
    StateVector state;
    int stat = orbit.interpolate(aztime, state, orbitMethod);
    if (stat != 0) {
        pyre::journal::error_t error("isce.geometry.Geometry.rdr2geo");
        error
            << pyre::journal::at(__HERE__)
            << "Error in getting state vector for bounds computation."
            << pyre::journal::newline
            << " - requested time: " << aztime << pyre::journal::newline
            << " - bounds: " << orbit.UTCtime[0] << " -> " << orbit.UTCtime[orbit.nVectors-1]
            << pyre::journal::endl;
    }

    // Setup geocentric TCN basis
    Basis TCNbasis;
    geocentricTCN(state, TCNbasis);

    // Compute satellite velocity magnitude
    const double vmag = LinAlg::norm(state.velocity());
    // Compute Doppler factor
    const double dopfact = 0.5 * wvl * doppler * slantRange / vmag;

    // Wrap range and Doppler factor in a Pixel object
    Pixel pixel(slantRange, dopfact, 0);

    // Finally, call rdr2geo
    stat = rdr2geo(pixel, TCNbasis, state, ellipsoid, demInterp, targetLLH, side,
                   threshold, maxIter, extraIter);
    return stat;
}


/** @param[in] pixel Pixel object
 * @param[in] TCNbasis Geocentric TCN basis corresponding to pixel
 * @param[in] state StateVector object
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] demInterp DEMInterpolator object
 * @param[out] targetLLH output Lon/Lat/Hae corresponding to pixel
 * @param[in] side +1 for left and -1 for right
 * @param[in] threshold Distance threshold for convergence
 * @param[in] maxIter Number of primary iterations
 * @param[in] extraIter Number of secondary iterations
 *
 * This is the elementary transformation from radar geometry to map geometry. The transformation is applicable for a single slant range and azimuth time (i.e., a single point target). The slant range and Doppler information are encapsulated in the Pixel object, so this function can work for both zero and native Doppler geometries. The azimuth time information is encapsulated in the TCNbasis and StateVector of the platform. For algorithmic details, see \ref overview_geometry "geometry overview".*/
int isce::geometry::
rdr2geo(const Pixel & pixel, const Basis & TCNbasis, const StateVector & state,
        const Ellipsoid & ellipsoid, const DEMInterpolator & demInterp,
        cartesian_t & targetLLH, int side, double threshold, int maxIter, int extraIter) {
    /*
    Assume orbit has been interpolated to correct azimuth time, then estimate geographic
    coordinates.
    */

    // Initialization
    cartesian_t targetVec, targetLLH_old, targetVec_old,
                lookVec, delta, delta_temp, vhat;

    // Compute normalized velocity
    LinAlg::unitVec(state.velocity(), vhat);

    // Unpack TCN basis vectors
    const cartesian_t that = TCNbasis.x0();
    const cartesian_t chat = TCNbasis.x1();
    const cartesian_t nhat = TCNbasis.x2();

    // Pre-compute TCN vector products
    const double ndotv = nhat[0]*vhat[0] + nhat[1]*vhat[1] + nhat[2]*vhat[2];
    const double vdott = vhat[0]*that[0] + vhat[1]*that[1] + vhat[2]*that[2];

    // Compute major and minor axes of ellipsoid
    const double major = ellipsoid.a();
    const double minor = major * std::sqrt(1.0 - ellipsoid.e2());

    // Set up orthonormal system right below satellite
    const double satDist = LinAlg::norm(state.position());
    const double eta = 1.0 / std::sqrt(
        std::pow(state.position()[0] / major, 2) +
        std::pow(state.position()[1] / major, 2) +
        std::pow(state.position()[2] / minor, 2)
    );
    const double radius = eta * satDist;
    const double hgt = (1.0 - eta) * satDist;

    // Iterate
    int converged = 0;
    double zrdr = targetLLH[2];
    for (int i = 0; i < (maxIter + extraIter); ++i) {

        // Near nadir test
        if ((hgt - zrdr) >= pixel.range())
            break;

        // Cache the previous solution
        targetLLH_old = targetLLH;

        // Compute angles
        const double a = satDist;
        const double b = radius + zrdr;
        const double costheta = 0.5 * (a / pixel.range() + pixel.range() / a 
                              - (b/a) * (b/pixel.range()));
        const double sintheta = std::sqrt(1.0 - costheta*costheta);
 
        // Compute TCN scale factors
        const double gamma = pixel.range() * costheta;
        const double alpha = (pixel.dopfact() - gamma * ndotv) / vdott;
        const double beta = -side * std::sqrt(std::pow(pixel.range(), 2)
                                            * std::pow(sintheta, 2) 
                                            - std::pow(alpha, 2));

        // Compute vector from satellite to ground
        LinAlg::linComb(alpha, that, beta, chat, delta_temp);
        LinAlg::linComb(1.0, delta_temp, gamma, nhat, delta);
        LinAlg::linComb(1.0, state.position(), 1.0, delta, targetVec);

        // Compute LLH of ground point
        ellipsoid.xyzToLonLat(targetVec, targetLLH);

        // Interpolate DEM at current lat/lon point
        targetLLH[2] = demInterp.interpolateLonLat(targetLLH[0], targetLLH[1]);

        // Convert back to XYZ with interpolated height
        ellipsoid.lonLatToXyz(targetLLH, targetVec);
        // Compute updated target height
        zrdr = LinAlg::norm(targetVec) - radius;

        // Check convergence
        LinAlg::linComb(1.0, state.position(), -1.0, targetVec, lookVec);
        const double rdiff = pixel.range() - LinAlg::norm(lookVec);
        if (std::abs(rdiff) < threshold) {
            converged = 1;
            break;
        // May need to perform extra iterations
        } else if (i > maxIter) {
            // XYZ position of old solution
            ellipsoid.lonLatToXyz(targetLLH_old, targetVec_old);
            // XYZ position of updated solution
            for (int idx = 0; idx < 3; ++idx)
                targetVec[idx] = 0.5 * (targetVec_old[idx] + targetVec[idx]);
            // Repopulate lat, lon, z
            ellipsoid.xyzToLonLat(targetVec, targetLLH);
            // Compute updated target height
            zrdr = LinAlg::norm(targetVec) - radius;
        }
    }

    // ----- Final computation: output points exactly at range pixel if converged

    // Compute angles
    const double a = satDist;
    const double b = radius + zrdr; 
    const double costheta = 0.5 * (a / pixel.range() + pixel.range() / a
                          - (b/a) * (b/pixel.range()));
    const double sintheta = std::sqrt(1.0 - costheta*costheta);

    // Compute TCN scale factors
    const double gamma = pixel.range() * costheta;
    const double alpha = (pixel.dopfact() - gamma * ndotv) / vdott;
    const double beta = -side * std::sqrt(std::pow(pixel.range(), 2)
                                        * std::pow(sintheta, 2)
                                        - std::pow(alpha, 2));

    // Compute vector from satellite to ground
    LinAlg::linComb(alpha, that, beta, chat, delta_temp);
    LinAlg::linComb(1.0, delta_temp, gamma, nhat, delta);
    LinAlg::linComb(1.0, state.position(), 1.0, delta, targetVec);

    // Compute LLH of ground point
    ellipsoid.xyzToLonLat(targetVec, targetLLH);

    // Return convergence flag
    return converged;
}


/** @param[in] inputLLH Lon/Lat/Hae of target of interest
 * @param[in] ellipsoid Ellipsoid object
 * @param[in] orbit Orbit object
 * @param[in] doppler   Poly2D Doppler model
 * @param[in] mode  ImageMode object
 * @param[out] aztime azimuth time of inputLLH w.r.t reference epoch of the orbit
 * @param[out] slantRange slant range to inputLLH
 * @param[in] threshold azimuth time convergence threshold in seconds
 * @param[in] maxIter Maximum number of Newton-Raphson iterations
 * @param[in] deltaRange step size used for computing derivative of doppler
 *
 * This is the elementary transformation from map geometry to radar geometry. The transformation is applicable for a single lon/lat/h coordinate (i.e., a single point target). For algorithmic details, see \ref overview_geometry "geometry overview".*/
int isce::geometry::
geo2rdr(const cartesian_t & inputLLH, const Ellipsoid & ellipsoid, const Orbit & orbit,
        const Poly2d & doppler, const ImageMode & mode, double & aztime, double & slantRange,
        double threshold, int maxIter, double deltaRange) {

    cartesian_t satpos, satvel, inputXYZ, dr;

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
        LinAlg::linComb(1.0, inputXYZ, -1.0, satpos, dr);
        slantRange = LinAlg::norm(dr);
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
        LinAlg::linComb(1.0, inputXYZ, -1.0, satpos, dr);
        slantRange = LinAlg::norm(dr);
        // Check convergence
        if (std::abs(slantRange - slantRange_old) < threshold) {
            converged = 1;
            return converged;
        } else {
            slantRange_old = slantRange;
        }

        // Compute slant range bin
        const double rbin = (slantRange - mode.startingRange()) / mode.rangePixelSpacing();
        // Compute doppler
        const double dopfact = LinAlg::dot(dr, satvel);
        const double fdop = doppler.eval(0, rbin) * dopscale;
        // Use forward difference to compute doppler derivative
        const double fdopder = (doppler.eval(0, rbin + deltaRange) * dopscale - fdop)
                             / deltaRange;
        
        // Evaluate cost function and its derivative
        const double fn = dopfact - fdop * slantRange;
        const double c1 = -1.0 * LinAlg::dot(satvel, satvel);
        const double c2 = (fdop / slantRange) + fdopder;
        const double fnprime = c1 + c2 * dopfact;

        // Update guess for azimuth time
        aztime -= fn / fnprime;
    }
    // If we reach this point, no convergence for specified threshold
    return converged;
}

// Utility function to compute geocentric TCN basis from state vector
void isce::geometry::
geocentricTCN(isce::core::StateVector & state, isce::core::Basis & basis) {
    // Compute basis vectors
    cartesian_t that, chat, nhat, temp;
    LinAlg::unitVec(state.position(), nhat);
    LinAlg::scale(nhat, -1.0);
    LinAlg::cross(nhat, state.velocity(), temp);
    LinAlg::unitVec(temp, chat);
    LinAlg::cross(chat, nhat, temp);
    LinAlg::unitVec(temp, that);
    // Store in basis object
    basis.x0(that);
    basis.x1(chat);
    basis.x2(nhat);
}


// end of file
