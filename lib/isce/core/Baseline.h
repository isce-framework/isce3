//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#pragma once

#include "Constants.h"
#include "Ellipsoid.h"
#include "forward.h"
#include "Metadata.h"
#include "Orbit.h"
#include "Vector.h"

namespace isce { namespace core {

/** Data structure for computing interferometric baselines */
class Baseline {
public:

    /** Get horizontal baseline */
    double horizontalBaseline() const { return _bh; }

    /** Get vertical baseline */
    double verticalBaseline() const { return _bv; }

    /** Get perpendicular baseline */
    double perpendicularBaseline() const { return (-1. * _bh * _coslook) + (_bv * _sinlook); }

    /** Get sin of look angle */
    double sinLook() const { return _sinlook;}

    /** Get cos of look angle */
    double cosLook() const { return _coslook;}

    /** Reference ECEF position for baseline */
    Vec3 refXyz() const { return _refxyz; }

    /** Unit vector in look direction */
    Vec3 look() const { return _look; }

    /** Unit vector in radial direction */
    Vec3 rhat() const { return _rhat; }

    /** Unit vector in cross track direction */
    Vec3 chat() const { return _chat; }

    /** Unit vector in direction of velocity */
    Vec3 vhat() const { return _vhat; }

    /** Return orbit interpolation method used */
    orbitInterpMethod orbitMethod() const { return _orbitMethod; }

    /** Return reference orbit */
    Orbit orbit1() const { return _orbit1; }

    /** Return secondary orbit */
    Orbit orbit2() const { return _orbit2; }

    /** Return metadata object */
    Metadata radar() const { return _radar; }

    /** Return ellipsoid*/
    Ellipsoid ellipsoid() const { return _elp; }

    /** Return magnitude of velocity */
    double velocityMagnitude() const { return _velocityMagnitude; }

    /** Initialization function to compute look vector and set basis vectors. */
    void init();

    /**
     * For a given time, calculate an orthogonal basis for cross-track and velocity directions for
     * orbit1.
     */
    void initBasis(double);

    /**
     * Given a position vector, calculate offset between reference position and that vector,
     * projected in the reference basis.
     */
    Vec3 calculateBasisOffset(const Vec3 &) const;

    /** Compute horizontal and vertical baselines. */
    void computeBaselines();

    /** Calculate look vector. */
    void calculateLookVector(double);

private:
    // Orbits
    Orbit _orbit1, _orbit2;
    // Metadata
    Metadata _radar;
    // Ellipsoid
    Ellipsoid _elp;
    // Orbit interpolatation method
    orbitInterpMethod _orbitMethod;
    // Basis vectors
    Vec3 _refxyz, _look, _rhat, _chat, _vhat;
    // Baseline scalars
    double _bh, _bv;
    // Look angle
    double _sinlook, _coslook;
    // Velocity magnitude
    double _velocityMagnitude;
};

}}

