// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018
//

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

int isce::geometry::
rdr2geo(double aztime, double slantRange, double dopfact, const Orbit & orbit,
        const Ellipsoid & ellipsoid, const DEMInterpolator & demInterp,
        cartesian_t & targetLLH, int side, double threshold, int maxIter, int extraIter,
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

    // Get TCN basis
    Basis TCNbasis;
    ellipsoid.TCNbasis(state.position(), state.velocity(), TCNbasis);

    // Convert satellite position to lat-lon
    cartesian_t llhsat;
    ellipsoid.xyzToLatLon(state.position(), llhsat);

    // Estimate heading
    const double heading = orbit.getENUHeading(aztime); 

    // Set peg point right below satellite
    isce::core::Peg peg(llhsat[0], llhsat[1], heading);
   
    // Initialize peg transformation
    Pegtrans ptm;
    ptm.radarToXYZ(ellipsoid, peg);

    // Wrap range and Doppler factor in a Pixel object
    Pixel pixel(slantRange, dopfact, 0);

    // Finally, call rdr2geo
    stat = rdr2geo(pixel, TCNbasis, state, ellipsoid, ptm, demInterp, targetLLH, side,
                   threshold, maxIter, extraIter);
    return stat;
}

int isce::geometry::
rdr2geo(const Pixel & pixel, const Basis & TCNbasis, const StateVector & state,
        const Ellipsoid & ellipsoid, const Pegtrans & ptm, const DEMInterpolator & demInterp,
        cartesian_t & targetLLH, int side, double threshold, int maxIter, int extraIter) {
    /*
    Assume orbit has been interpolated to correct azimuth time, then estimate geographic
    coordinates.
    */

    // Initialization
    cartesian_t targetSCH, targetVec, targetLLH_old, targetVec_old,
                lookVec, delta, delta_temp, vhat, satLLH;
    const double degrees = 180.0 / M_PI;

    // Compute normalized velocity
    LinAlg::unitVec(state.velocity(), vhat);
    targetSCH[2] = targetLLH[2];

    // Unpack TCN basis vectors
    const cartesian_t that = TCNbasis.x0();
    const cartesian_t chat = TCNbasis.x1();
    const cartesian_t nhat = TCNbasis.x2();

    // Pre-compute TCN vector products
    const double ndotv = nhat[0]*vhat[0] + nhat[1]*vhat[1] + nhat[2]*vhat[2];
    const double vdott = vhat[0]*that[0] + vhat[1]*that[1] + vhat[2]*that[2];

    // Compute satellite LLH to get height above ellipsoid
    ellipsoid.xyzToLatLon(state.position(), satLLH);

    // Iterate
    int converged = 0;
    for (int i = 0; i < (maxIter + extraIter); ++i) {

        // Cache the previous solution
        targetLLH_old = targetLLH;

        // Compute angles
        const double a = satLLH[2] + ptm.radcur;
        const double b = ptm.radcur + targetSCH[2];
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
        ellipsoid.xyzToLatLon(targetVec, targetLLH);

        // Interpolate DEM at current lat/lon point
        targetLLH[2] = demInterp.interpolate(degrees*targetLLH[0], degrees*targetLLH[1]);
        // Convert back to XYZ with interpolated height
        ellipsoid.latLonToXyz(targetLLH, targetVec);
        // Compute updated SCH coordinates
        ptm.convertSCHtoXYZ(targetSCH, targetVec, isce::core::XYZ_2_SCH);

        // Check convergence
        LinAlg::linComb(1.0, state.position(), -1.0, targetVec, lookVec);
        const double rdiff = pixel.range() - LinAlg::norm(lookVec);
        if (std::abs(rdiff) < threshold) {
            converged = 1;
            break;
        // May need to perform extra iterations
        } else if (i > maxIter) {
            // XYZ position of old solution
            ellipsoid.latLonToXyz(targetLLH_old, targetVec_old);
            // XYZ position of updated solution
            for (int idx = 0; idx < 3; ++idx)
                targetVec[idx] = 0.5 * (targetVec_old[idx] + targetVec[idx]);
            // Repopulate lat, lon, z
            ellipsoid.xyzToLatLon(targetVec, targetLLH);
            // Recompute SCH coordinates
            ptm.convertSCHtoXYZ(targetSCH, targetVec, isce::core::XYZ_2_SCH);
        }
    }

    // ----- Final computation: output points exactly at range pixel if converged

    // Compute angles
    const double a = satLLH[2] + ptm.radcur;
    const double b = ptm.radcur + targetSCH[2];
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
    ellipsoid.xyzToLatLon(targetVec, targetLLH);    

    // Return convergence flag
    return converged;
}

int isce::geometry::
geo2rdr(const cartesian_t & inputLLH, const Ellipsoid & ellipsoid, const Orbit & orbit,
        const Poly2d & doppler, const Metadata & meta, double & aztime, double & slantRange,
        double threshold, int maxIter, double deltaRange) {

    cartesian_t satpos, satvel, inputXYZ, dr;

    // Convert LLH to XYZ
    ellipsoid.latLonToXyz(inputLLH, inputXYZ);

    // Pre-compute scale factor for doppler
    const double dopscale = 0.5 * meta.radarWavelength;

    // Compute minimum and maximum valid range
    const double rangeMin = meta.rangeFirstSample;
    const double rangeMax = rangeMin + meta.slantRangePixelSpacing * (meta.width - 1);

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
        const double rbin = (slantRange - meta.rangeFirstSample) / meta.slantRangePixelSpacing;
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

// end of file
