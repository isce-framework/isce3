//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include "Baseline.h"
#include "Constants.h"
#include "Peg.h"
#include "Pegtrans.h"
using isce::core::Baseline;
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
    // Interpolate orbit to azimuth time
    cartesian_t xyz, vel;
    _orbit1.interpolate(t, xyz, vel, _orbitMethod);
    _refxyz = xyz;
    _velocityMagnitude = vel.norm();
    // Get normalized vectors
    const Vec3 vel_norm = vel / _velocityMagnitude;
    _rhat = xyz.unitVec();
    // Compute cross-track vectors
    _chat = _rhat.cross(vel_norm).unitVec();
    // Compute velocity vector perpendicular to cross-track vector
    _vhat = _chat.cross(_rhat).unitVec();
}

isce::core::cartesian_t Baseline::calculateBasisOffset(const cartesian_t &position) const {
    /*
     * Given a position vector, calculate offset between reference position and that vector,
     * projected in the reference basis.
     */
    const Vec3 dx = position - _refxyz;
    return Vec3 { dx.dot(_vhat),
                  dx.dot(_rhat),
                  dx.dot(_chat) };
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
    _elp.xyzToLonLat(xyz, llh);
    // Make a Peg
    Peg peg(llh[1], llh[0], _radar.pegHeading);
    // And a Peg Transformation
    Pegtrans ptm;
    ptm.radarToXYZ(_elp, peg);
    double Ra = ptm.radcur;
    double height = llh[2];
    double R0 = _radar.rangeFirstSample;
    _coslook = (height * ((2. * Ra) + height) + (R0 * R0)) / (2. * R0 * (Ra + height));
    _sinlook = sqrt(1. - (_coslook * _coslook));
}
