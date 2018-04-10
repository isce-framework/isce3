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
    _orbitMethod = HERMITE_METHOD;
    // Initialize basis for the first orbit using the middle of the orbit
    cartesian_t _1, _2;
    double tmid;
    _orbit1.getStateVector(_orbit1.nVectors/2, tmid, _1, _2);
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
    cartesian_t xyz, vel, crossvec, vertvec, vel_norm;
    // Interpolate orbit to azimuth time
    _orbit1.interpolate(t, xyz, vel, _orbitMethod);
    _refxyz = xyz;
    _velocityMagnitude = LinAlg::norm(vel);
    // Get normalized vectors
    LinAlg::unitVec(xyz, _rhat);
    LinAlg::unitVec(vel, vel_norm);
    // Compute cross-track vectors
    LinAlg::cross(_rhat, vel_norm, crossvec);
    LinAlg::unitVec(crossvec, _chat);
    // Compute velocity vector perpendicular to cross-track vector
    LinAlg::cross(_chat, _rhat, vertvec);
    LinAlg::unitVec(vertvec, _vhat);
}

isce::core::cartesian_t Baseline::calculateBasisOffset(const cartesian_t &position) const {
    /*
     * Given a position vector, calculate offset between reference position and that vector,
     * projected in the reference basis.
     */
    cartesian_t dx = {position[0] - _refxyz[0],
                      position[1] - _refxyz[1],
                      position[2] - _refxyz[2]};
    cartesian_t off = {LinAlg::dot(dx, _vhat),
                       LinAlg::dot(dx, _rhat),
                       LinAlg::dot(dx, _chat)};
    return off;
}

void Baseline::computeBaselines() {
    /*
     * Compute horizontal and vertical baselines.
     */
    cartesian_t xyz2, vel2, offset;
    double t, delta_t;
    // Start with sensing mid of orbit 2
    _orbit2.getStateVector(_orbit2.nVectors/2, t, xyz2, vel2);
    for (int iter=0; iter<2; ++iter) {
        //Interpolate orbit to azimuth time
        _orbit2.interpolate(t, xyz2, vel2, _orbitMethod);
        // Compute adjustment to slave time
        offset = calculateBasisOffset(xyz2);
        delta_t = offset[0] / _velocityMagnitude;
        t -= delta_t;
    }
    _bh = offset[2];
    _bv = offset[1];
}

void Baseline::calculateLookVector(double t) {
    /*
     * Calculate look vector.
     */
    // Local working vectors
    cartesian_t xyz, vel, llh;
    // Interpolate orbit to azimuth time
    _orbit1.interpolate(t, xyz, vel, _orbitMethod);
    _elp.xyzToLatLon(xyz, llh);
    // Make a Peg
    Peg peg(llh[0], llh[1], _radar.pegHeading);
    // And a Peg Transformation
    Pegtrans ptm;
    ptm.radarToXYZ(_elp, peg);
    double Ra = ptm.radcur;
    double height = llh[2];
    double R0 = _radar.rangeFirstSample;
    _coslook = (height * ((2. * Ra) + height) + (R0 * R0)) / (2. * R0 * (Ra + height));
    _sinlook = sqrt(1. - (_coslook * _coslook));
}
