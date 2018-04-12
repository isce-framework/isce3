// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

// isce::core
#include <isce/core/LinAlg.h>

// isce::geometry
#include "Geometry.h"

// pull in useful isce::core namespace
using isce::core::LinAlg;
using isce::core::Ellipsoid;
using isce::core::Pegtrans;
using isce::core::StateVector;

int isce::geometry::Geometry::
rdr2geo(const Pixel & pixel, const Basis & basis, const StateVector & state,
        const Ellipsoid & ellipsoid, const Pegtrans & ptm, const DEMInterpolator & demInterp,
        cartesian_t & targetLLH, int side, double threshold, int maxIter, int extraIter) {
    /*
    - Assume orbit has been interpolated to correct azimuth time. Consider putting in a
      check for this condition.

    - Start with position and velocity of spacecraft
    */

    // Initialization
    cartesian_t targetSCH, targetVec, targetLLH_old, targetVec_old,
                lookVec, delta, delta_temp;
    targetSCH[2] = targetLLH[2];

    // Iterate
    int converged = 0;
    for (int i = 0; i < (maxIter + extraIter); ++i) {

        // Cache the previous solution
        targetLLH_old = targetLLH;

        // Compute angles
        const double a = targetLLH[2] + ptm.radcur;
        const double b = ptm.radcur + targetSCH[2];
        const double costheta = 0.5 * (a / pixel.range() + pixel.range() / a 
                              - (b/a) * (b/pixel.range()));
        const double sintheta = std::sqrt(1.0 - costheta*costheta);

        // Compute TCN scale factors
        const double gamma = pixel.range() * costheta;
        const double alpha = pixel.dopfact() - gamma*LinAlg::dot(basis.nhat(), state.velocity()) 
                           / LinAlg::dot(state.velocity(), basis.that());
        const double beta = -side * std::sqrt(std::pow(pixel.range(), 2)
                                            * std::pow(sintheta, 2) 
                                            - std::pow(alpha, 2));

        // Compute vector from satellite to ground
        LinAlg::linComb(alpha, basis.that(), beta, basis.chat(), delta_temp);
        LinAlg::linComb(1.0, delta_temp, gamma, basis.nhat(), delta);
        LinAlg::linComb(1.0, state.position(), 1.0, delta, targetVec);

        // Compute LLH of ground point
        ellipsoid.xyzToLatLon(targetVec, targetLLH);

        // Interpolate DEM at current lat/lon point
        targetLLH[2] = demInterp.interpolate(targetLLH[0], targetLLH[1]);
        // Convert back to XYZ with interpolated height
        ellipsoid.latLonToXyz(targetLLH, targetVec);
        // Compute updated SCH coordinates
        ptm.convertSCHtoXYZ(targetSCH, targetVec, isce::core::XYZ_2_SCH);

        // Check convergence
        LinAlg::linComb(1.0, state.position(), -1.0, targetVec, lookVec);
        const double rdiff = pixel.range() - LinAlg::norm(lookVec);
        if (std::abs(rdiff) < threshold) {
            converged = 1;
            return converged;
        // May need to perform extra iterations
        } else if (i > maxIter) {
            // XYZ position of old solution
            ellipsoid.latLonToXyz(targetLLH_old, targetVec_old);
            // XYZ position of updated solution
            for (int idx = 0; idx < 3; ++idx)
                targetVec[idx] = 0.5 * (targetVec_old[idx] + targetVec[idx]);
            // Repopulate lat, lon, z
            ellipsoid.xyzToLatLon(targetVec, targetLLH);
            // Check convergence
            LinAlg::linComb(1.0, state.position(), -1.0, targetVec, lookVec);
            const double rdiff = pixel.range() - LinAlg::norm(lookVec);
            if (std::abs(rdiff) < threshold) {
                converged = 1;
                return converged;
            }
        }
    }
    // If we reach this point, no convergence for specified threshold
    return converged;
}


//void isce::core::Geometry::
//geo2rdr() {}

// end of file
