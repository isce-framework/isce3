//-*- C++ -*-
//-*- coding: utf-8 -*-

#include <iostream>
#include <typeinfo>
#include <exception>
#include "Constants.h"
#include "LinAlg.h"
#include "Peg.h"
#include "Doppler.h"

// Doppler constructor
template<class Attitude>
isce::core::Doppler<Attitude>::
Doppler(Orbit * orbit, Attitude * attitude, Ellipsoid * ellipsoid, double epoch) {

    // Initialize state vectors
    satxyz = {0.0, 0.0, 0.0};
    satvel = {0.0, 0.0, 0.0};
    satllh = {0.0, 0.0, 0.0};

    // Interpolate orbit to epoch
    int stat = orbit->interpolate(epoch, satxyz, satvel, HERMITE_METHOD);
    if (stat != 0) {
        std::cerr << "Error in Doppler::Doppler - error getting state vector." << std::endl;
        std::cerr << " - requested time: " << epoch << std::endl;
        std::cerr << " - bounds: " << orbit->UTCtime[0] << " -> " 
                  << orbit->UTCtime[orbit->nVectors-1] << std::endl;
        throw std::out_of_range("Orbit out of range");
    }
    // Compute llh
    ellipsoid->xyzToLatLon(satxyz, satllh);
    // Compute heading
    double heading = orbit->getENUHeading(epoch);

    // Create a temporary peg object
    Peg peg(satllh[0], satllh[1], heading);

    // Set SCH information
    ptm.radarToXYZ(*ellipsoid, peg);

    // Save objects
    this->orbit = orbit;
    this->attitude = attitude;
    this->ellipsoid = ellipsoid;
    this->epoch = epoch;

}

// Evaluate Doppler centroid at a specific slant range
template<class Attitude>
double isce::core::Doppler<Attitude>::
centroid(double slantRange, double wvl, std::string frame, size_t max_iter,
    int side, bool precession) {

    // Compute ECI velocity if attitude angles are provided in inertial frame
    std::vector<double> Va(3);
    if (frame == "inertial") {
        std::vector<double> w{0.0, 0.0, 0.00007292115833};
        LinAlg::cross(w, satxyz, Va);
        for (size_t i = 0; i < 3; ++i) {
            Va[i] += satvel[i];
        }
    } else {
        Va = satvel;
    }

    // Compute u0 directly if quaternion
    vector_t u0(3), temp(3);
    if (typeid(attitude) == typeid(Quaternion *)) {
        
        temp = {1.0, 0.0, 0.0};
        matrix_t R = attitude->rotmat("");
        LinAlg::matVec(R, temp, u0); 

    // Else multiply orbit and attitude matrix
    } else {

        // Compute vectors for TCN-like basis
        vector_t q(3), c(3), b(3), a(3);
        temp = {satxyz[0], satxyz[1], satxyz[2] / (1 - ellipsoid->e2)};
        LinAlg::unitVec(temp, q);
        c = {-q[0], -q[1], -q[2]};
        LinAlg::cross(c, Va, temp);
        LinAlg::unitVec(temp, b);
        LinAlg::cross(b, c, a);

        // Stack basis vectors to get orbit matrix
        matrix_t L0(3, std::vector<double>(3, 0.0));
        for (size_t i = 0; i < 3; ++i) {
            L0[i][0] = a[i];
            L0[i][1] = b[i];
            L0[i][2] = c[i];
        }

        // Get attitude matrix
        matrix_t L = attitude->rotmat("ypr");

        // Compute u0
        u0 = {1.0, 0.0, 0.0};
        LinAlg::matVec(L, u0, temp);
        LinAlg::matVec(L0, temp, u0);
    }

    // Fake the velocity vector by using u0 scaled by absolute velocity
    double vmag = LinAlg::norm(Va);
    vector_t vel = {u0[0] * vmag, u0[1] * vmag, u0[2] * vmag};

    // Set up TCN basis
    vector_t that(3), chat(3), nhat(3), vhat(3);
    ellipsoid->TCNbasis(satxyz, vel, that, chat, nhat);
    LinAlg::unitVec(vel, vhat);

    // Iterate
    vector_t targetVec(3), targetSCH(3), targetLLH(3), delta(3), lookVec(3);
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
        ellipsoid->xyzToLatLon(targetVec, targetLLH);
        // Set the expected target height
        targetLLH[2] = height;
        // Compute updated sch height
        ellipsoid->latLonToXyz(targetLLH, targetVec);
        ptm.convertSCHtoXYZ(targetSCH, targetVec, XYZ_2_SCH);
        zsch = targetSCH[2];

        // Check convergence
        LinAlg::linComb(1.0, satxyz, -1.0, targetVec, lookVec);
        double rdiff = slantRange - LinAlg::norm(lookVec);
        if (std::abs(rdiff) < 1.0e-8)
            break;
    }

    // Compute unitary look vector
    vector_t R(3), Rhat(3);
    ellipsoid->latLonToXyz(targetLLH, targetVec);
    LinAlg::linComb(1.0, satxyz, -1.0, targetVec, R);
    LinAlg::unitVec(R, Rhat);
    
    // Compute doppler
    double fd = -2.0 / wvl * LinAlg::dot(satvel, Rhat);
    return fd;

}

    

// end of file
