//-*- C++ -*-
//-*- coding: utf-8 -*-

// Standard
#include <iostream>

// isce::core
#include "Baseline.h"
#include "Peg.h"
#include "Pegtrans.h"


// Empty constructor
isce::core::Baseline::Baseline() {
}


// Initialization function to compute look vector and set basis vectors
void isce::core::Baseline::init() {

    // Set orbit method
    orbit_method = HERMITE_METHOD;

    // Initialize basis for the first orbit using the middle of the orbit
    int index_mid = orbit1.nVectors / 2;
    double tmid = orbit1.UTCtime[index_mid];
    initBasis(tmid);

    // Use radar metadata to compute look vector at midpoint
    _calculateLookVector(tmid);

}


// For a given time, calculate an orthogonal basis for cross-track and velocity
// directions for orbit1
void isce::core::Baseline::initBasis(double t) {

    // Local working vectors
    std::vector<double> xyz(3), vel(3), crossvec(3), vertvec(3), vel_norm(3);

    // Interpolate orbit to azimuth time
    orbit1.interpolate(t, xyz, vel, orbit_method);
    _refxyz = xyz;
    _velocityMagnitude = linalg.norm(vel);

    // Get normalized vectors
    linalg.unitVec(xyz, _rhat);
    linalg.unitVec(vel, vel_norm);

    // Compute cross-track vector
    linalg.cross(_rhat, vel_norm, crossvec);
    linalg.unitVec(crossvec, _chat);

    // Compute velocity vector perpendicular to cross-track vector
    linalg.cross(_chat, _rhat, vertvec);
    linalg.unitVec(vertvec, _vhat); 

}


// Given a position vector, calculate offset between reference position
// and that vector, projected in the reference basis
std::vector<double> isce::core::Baseline::calculateBasisOffset(
    std::vector<double> & position) {

    std::vector<double> dx(3), off(3);

    // Compute difference
    for (int i = 0; i < 3; ++i) {
        dx[i] = position[i] - _refxyz[i];
    }

    // Project offset onto velocity vector
    off[0] = linalg.dot(dx, _vhat);

    // Project offset onto position vector
    off[1] = linalg.dot(dx, _rhat);

    // Project offset onto cross-track vector
    off[2] = linalg.dot(dx, _chat);

    return off;

}

// Compute horizontal and vertical baselines
void isce::core::Baseline::computeBaselines() {

    std::vector<double> xyz2(3), vel2(3), offset(3);
    double t, delta_t;

    // Start with sensing mid of orbit 2
    int index_mid = orbit2.nVectors / 2;
    t = orbit2.UTCtime[index_mid];

    for (int iter = 0; iter < 2; ++iter) {

        // Interpolate orbit to azimuth time
        orbit2.interpolate(t, xyz2, vel2, orbit_method);
   
        // Compute adjustment to slave time
        offset = calculateBasisOffset(xyz2);
        delta_t = offset[0] / _velocityMagnitude;
        t -= delta_t;
    }
     
    _bh = offset[2];
    _bv = offset[1];

}


// Calculate look vector
void isce::core::Baseline::_calculateLookVector(double t) {

    // Local working vectors
    std::vector<double> xyz(3), vel(3), llh(3);
    
    // Interpolate orbit to azimuth time
    orbit1.interpolate(t, xyz, vel, orbit_method);
    ellp.xyzToLatLon(xyz, llh);

    // Make a peg
    Peg peg;
    peg.lat = llh[0];
    peg.lon = llh[1];
    peg.hdg = radar.pegHeading;

    // And a peg transformation
    Pegtrans ptm;
    ptm.radarToXYZ(ellp, peg);

    double Ra = ptm.radcur;
    double height = llh[2];
    double R0 = radar.rangeFirstSample;

    _coslook = (height * (2.0*Ra + height) + R0*R0) / (2*R0*(Ra + height));
    _sinlook = std::sqrt(1.0 - _coslook * _coslook);

}

// end of file
