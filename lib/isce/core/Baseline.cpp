//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <vector>
#include "Baseline.h"
#include "Constants.h"
#include "LinAlg.h"
#include "Peg.h"
#include "Pegtrans.h"
using std::vector;
using isce::core::Baseline;
using isce::core::LinAlg;
using isce::core::orbitInterpMethod;
using isce::core::HERMITE_METHOD;
using isce::core::Peg;
using isce::core::Pegtrans;

void Baseline::init() {
    /*
     * Initialization function to compute look vector and set basis vectors.
     */
    // Set orbit method
    orbit_method = HERMITE_METHOD;
    // Initialize basis for the first orbit using the middle of the orbit
    vector<double> _1(3), _2(3);
    double tmid;
    orbit1.getStateVector(orbit1.nVectors/2, tmid, _1, _2);
    initBasis(tmid);
    // Use radar metadata to compute look vector at midpoint
    calculateLookVector(tmid);
}

void Baseline::initBasis(double t) {
    /*
     * For a given time, calculate an orthogonal basis for cross-track and velocity directions for
     * orbit1.
     */
    // Local working vectors
    vector<double> xyz(3), vel(3), crossvec(3), vertvec(3), vel_norm(3);
    // Interpolate orbit to azimuth time
    orbit1.interpolate(t, xyz, vel, orbit_method);
    refxyz = xyz;
    velocityMagnitude = LinAlg::norm(vel);
    // Get normalized vectors
    LinAlg::unitVec(xyz, rhat);
    LinAlg::unitVec(vel, vel_norm);
    // Compute cross-track vectors
    LinAlg::cross(rhat, vel_norm, crossvec);
    LinAlg::unitVec(crossvec, chat);
    // Compute velocity vector perpendicular to cross-track vector
    LinAlg::cross(chat, rhat, vertvec);
    LinAlg::unitVec(vertvec, vhat);
}

vector<double> Baseline::calculateBasisOffset(const vector<double> &position) const {
    /*
     * Given a position vector, calculate offset between reference position and that vector,
     * projected in the reference basis.
     */
    vector<double> dx = {position[0] - refxyz[0],
                         position[1] - refxyz[1],
                         position[2] - refxyz[2]};
    vector<double> off = {LinAlg::dot(dx, vhat),
                          LinAlg::dot(dx, rhat),
                          LinAlg::dot(dx, chat)};
    return off;
}

void Baseline::computeBaselines() {
    /*
     * Compute horizontal and vertical baselines.
     */
    vector<double> xyz2(3), vel2(3), offset(3);
    double t, delta_t;
    // Start with sensing mid of orbit 2
    orbit2.getStateVector(orbit2.nVectors/2, t, xyz2, vel2);
    for (int iter=0; iter<2; ++iter) {
        //Interpolate orbit to azimuth time
        orbit2.interpolate(t, xyz2, vel2, orbit_method);
        // Compute adjustment to slave time
        offset = calculateBasisOffset(xyz2);
        delta_t = offset[0] / velocityMagnitude;
        t -= delta_t;
    }
    bh = offset[2];
    bv = offset[1];
}

void Baseline::calculateLookVector(double t) {
    /*
     * Calculate look vector.
     */
    // Local working vectors
    vector<double> xyz(3), vel(3), llh(3);
    // Interpolate orbit to azimuth time
    orbit1.interpolate(t, xyz, vel, orbit_method);
    elp.xyzToLatLon(xyz, llh);
    // Make a Peg
    Peg peg(llh[0], llh[1], radar.pegHeading);
    // And a Peg Transformation
    Pegtrans ptm;
    ptm.radarToXYZ(elp, peg);
    double Ra = ptm.radcur;
    double height = llh[2];
    double R0 = radar.rangeFirstSample;
    coslook = (height * ((2. * Ra) + height) + (R0 * R0)) / (2. * R0 * (Ra + height));
    sinlook = sqrt(1. - (coslook * coslook));
}
