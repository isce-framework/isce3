// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017-2018
//

#include "Geometry.h"
#include "LinAlg.h"

void isce::core::Geometry::
rdr2geo(double slantRange, double dopfact, const cartesian_t & that,
        const cartesian_t & chat, const cartesian_t & nhat,
        Ellipsoid & ellipsoid, Pegtrans & ptm, DEMInterpolator & demInterp,
        cartesian_t & targetLLH, 
    /*
    - Assume orbit has been interpolated to correct azimuth time. Consider putting in a
      check for this condition.

    - Start with position and velocity of spacecraft
    */

    cartesian_t targetLLH_old, targetVec_old, targetSCH;

    // Initialization
    targetSCH[2] = targetLLH[2];

    // Iterate
    for (size_t i = 0; i < (maxIter + extraIter); ++i) {

        // Cache the previous solution
        targetLLH_old = targetLLH;

        // Compute angles
        const double a = targetLLH[2] + radius;
        const double b = radius + targetSCH[2];
        const double costheta = 0.5*(a/slantRange + slantRange/a - (b/a)*(b/slantRange));
        const double sintheta = std::sqrt(1.0 - costheta*costheta);

        // Compute TCN scale factors
        const double gamma = slantRange * costheta;
        const double alpha = dopfact - gamma*LinAlg::dot(nhat,satVel) / LinAlg::dot(satVel,that);
        const double beta = -side*std::sqrt(slantRange*slantRange*sintheta*sintheta - alpha*alpha);

        // Compute vector from satellite to ground
        LinAlg::linComb(alpha, that, beta, chat, delta_temp);
        LinAlg::linComb(1.0, delta_temp, gamma, nhat, delta);
        LinAlg::linComb(1.0, satPos, 1.0, delta, targetVec);

        // Compute LLH of ground point
        ellipsoid.xyzToLatLon(targetVec, targetLLH);

        // Interpolate DEM at current lat/lon point
        float height = demInterp.interpolate(targetLLH[0], targetLLH[1]);
        targetLLH[2] = height;
        // Convert back to XYZ with interpolated height
        ellipsoid.latLonToXyz(targetLLH, targetVec);
        // Compute updated SCH coordinates
        ptm.convertSCHtoXYZ(targetSCH, targetVec, XYZ_2_SCH);

        // Check convergence
        LinAlg::linComb(1.0, satPos, -1.0, targetVec, lookVec);
        const double rdiff = slantRange - LinAlg::norm(lookVec);
        if (std::abs(rdiff) < threshold) {
            return;
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
            LinAlg::linComb(1.0, satPos, -1.0, targetVec, loookVec);
            const double rdiff = slantRange - LinAlg::norm(lookVec);
            if (std::abs(rdiff) < threshold) {
                return;
            }
        }
    }
}


void isce::core::Geometry::
geo2rdr() {


}

// end of file
