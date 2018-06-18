//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018
//

#include <iostream>
#include <exception>
#include "Constants.h"
#include "LinAlg.h"
#include "Peg.h"
#include "Doppler.h"

/** @param[in] orbit Orbit data structure
 *  @param[in] attitude Attitude data structure
 *  @param[in] ellipsoid Ellipsoid data structure used for orbit and attitude representation
 *  @param[in] epoch Epoch time of interest*/
isce::core::Doppler::
Doppler(Orbit & orbit, Attitude * attitude, Ellipsoid & ellipsoid, double epoch) {

    // Perform check that input attitude is supported
    AttitudeType atype = attitude->attitudeType();
    if (atype != QUATERNION_T && atype != EULERANGLES_T) {
        throw std::invalid_argument("Unsupported attitude object.");
    }

    // Initialize state vectors
    satxyz = {0.0, 0.0, 0.0};
    satvel = {0.0, 0.0, 0.0};
    satllh = {0.0, 0.0, 0.0};

    // Interpolate orbit to epoch
    int stat = orbit.interpolate(epoch, satxyz, satvel, HERMITE_METHOD);
    if (stat != 0) {
        std::cerr << "Error in Doppler::Doppler - error getting state vector." << std::endl;
        std::cerr << " - requested time: " << epoch << std::endl;
        std::cerr << " - bounds: " << orbit.UTCtime[0] << " -> " 
                  << orbit.UTCtime[orbit.nVectors-1] << std::endl;
        throw std::out_of_range("Orbit out of range");
    }
    // Compute llh
    ellipsoid.xyzToLonLat(satxyz, satllh);
    // Compute heading
    double heading = orbit.getENUHeading(epoch);

    // Create a temporary peg object
    Peg peg(satllh[1], satllh[0], heading);

    // Set SCH information
    ptm.radarToXYZ(ellipsoid, peg);

    // Save objects
    this->orbit = orbit;
    this->ellipsoid = ellipsoid;
    this->epoch = epoch;
    // Cast the attitude pointer to an Attitude pointer
    this->attitude = static_cast<Attitude *>(attitude);

}

/**@param[in] slantRange slant range to pixel of interest in meters
 * @param[in] wvl Wavelength of imaging platform in meters
 * @param[in] frame Can be "inertial" or "fixed"
 * @param[in] max_iter Number of iterations. Default is 10.
 * @param[in] side -1 for Right and +1 for left. Default is -1.
 * @param[in] precession To apply precession correction or not*/
double isce::core::Doppler::
centroid(double slantRange, double wvl, std::string frame, size_t max_iter,
    int side, bool precession) {

    // Compute ECI velocity if attitude angles are provided in inertial frame
    cartesian_t Va;
    if (frame.compare("inertial") == 0) {
        cartesian_t w{0.0, 0.0, 0.00007292115833};
        LinAlg::cross(w, satxyz, Va);
        for (size_t i = 0; i < 3; ++i) {
            Va[i] += satvel[i];
        }
    } else {
        Va = satvel;
    }

    // Compute u0 directly if quaternion
    cartesian_t u0, temp;
    if (attitude->attitudeType() == QUATERNION_T) {
        
        temp = {1.0, 0.0, 0.0};
        cartmat_t R = attitude->rotmat("");
        LinAlg::matVec(R, temp, u0); 

    // Else multiply orbit and attitude matrix
    } else if (attitude->attitudeType() == EULERANGLES_T) {

        // Compute vectors for TCN-like basis
        cartesian_t q, c, b, a;
        if (attitude->yawOrientation().compare("normal") == 0) {
            temp = {std::cos(satllh[1]) * std::cos(satllh[0]),
                    std::cos(satllh[1]) * std::sin(satllh[0]),
                    std::sin(satllh[1])};
        } else if (attitude->yawOrientation().compare("center") == 0) {
            temp = {satxyz[0], satxyz[1], satxyz[2]};
        }
        LinAlg::unitVec(temp, q);
        c = {-q[0], -q[1], -q[2]};
        LinAlg::cross(c, Va, temp);
        LinAlg::unitVec(temp, b);
        LinAlg::cross(b, c, a);

        // Stack basis vectors to get orbit matrix
        cartmat_t L0;
        for (size_t i = 0; i < 3; ++i) {
            L0[i][0] = a[i];
            L0[i][1] = b[i];
            L0[i][2] = c[i];
        }

        // Get attitude matrix
        cartmat_t L = attitude->rotmat("ypr");

        // Compute u0
        u0 = {1.0, 0.0, 0.0};
        LinAlg::matVec(L, u0, temp);
        LinAlg::matVec(L0, temp, u0);
    }

    // Fake the velocity vector by using u0 scaled by absolute velocity
    double vmag = LinAlg::norm(Va);
    cartesian_t vel = {u0[0] * vmag, u0[1] * vmag, u0[2] * vmag};

    // Set up TCN basis
    cartesian_t that, chat, nhat, vhat;
    ellipsoid.TCNbasis(satxyz, vel, that, chat, nhat);
    LinAlg::unitVec(vel, vhat);

    // Iterate
    cartesian_t targetVec, targetSCH, targetLLH, delta, lookVec;
    double height = 0.0;
    double zsch = height;
    double dopfact = 0.0;
    for (size_t i = 0; i < max_iter; ++i) {

        // Compute angles
        double a = satllh[2] + ptm.radcur;
        double b = ptm.radcur + zsch;
        double costheta = 0.5*(a/slantRange + slantRange/a - (b/a)*(b/slantRange));
        double sintheta = std::sqrt(1.0 - costheta*costheta);

        // Compute TCN scale factors
        double gamma = slantRange * costheta;
        double alpha = dopfact - gamma*LinAlg::dot(nhat,vel) / LinAlg::dot(vel,that);
        double beta = -side*std::sqrt(slantRange*slantRange*sintheta*sintheta - alpha*alpha);

        // Compute vector from satellite to ground
        LinAlg::linComb(alpha, that, beta, chat, temp);
        LinAlg::linComb(1.0, temp, gamma, nhat, delta);
        LinAlg::linComb(1.0, satxyz, 1.0, delta, targetVec);

        // Compute LLH of ground point
        ellipsoid.xyzToLonLat(targetVec, targetLLH);
        // Set the expected target height
        targetLLH[2] = height;
        // Compute updated sch height
        ellipsoid.lonLatToXyz(targetLLH, targetVec);
        ptm.convertSCHtoXYZ(targetSCH, targetVec, XYZ_2_SCH);
        zsch = targetSCH[2];

        // Check convergence
        LinAlg::linComb(1.0, satxyz, -1.0, targetVec, lookVec);
        double rdiff = slantRange - LinAlg::norm(lookVec);
        if (std::abs(rdiff) < 1.0e-8)
            break;
    }

    // Compute unitary look vector
    cartesian_t R, Rhat;
    ellipsoid.lonLatToXyz(targetLLH, targetVec);
    LinAlg::linComb(1.0, satxyz, -1.0, targetVec, R);
    LinAlg::unitVec(R, Rhat);
    
    // Compute doppler
    double fd = -2.0 / wvl * LinAlg::dot(satvel, Rhat);
    return fd;
}

// end of file
