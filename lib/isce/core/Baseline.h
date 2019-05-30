//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CORE_BASELINE_H
#define ISCE_CORE_BASELINE_H

#include "Cartesian.h"
#include "Constants.h"
#include "Ellipsoid.h"
#include "Metadata.h"
#include "Orbit.h"

// Declaration
namespace isce {
    namespace core {
        class Baseline;
    }
}

/** Data structure for computing interferometric baselines */
class isce::core::Baseline {

    public:
        /** Empty constructor */
        Baseline() {}

        /** Copy constructor */
        Baseline(const Baseline &b) : 
            _orbit1(b.orbit1()), _orbit2(b.orbit2()), _radar(b.radar()), _elp(b.ellipsoid()),
            _orbitMethod(b.orbitMethod()), _refxyz(b.refXyz()), _look(b.look()),
            _rhat(b.rhat()), _chat(b.chat()), _vhat(b.vhat()), _bh(b.horizontalBaseline()),
            _bv(b.verticalBaseline()), _sinlook(b.sinLook()),
            _coslook(b.cosLook()), _velocityMagnitude(b.velocityMagnitude()) {}

        /** Equality comparison operator */
        inline Baseline& operator=(const Baseline&);

        /** Get horizontal baseline */
        double horizontalBaseline() const { return _bh; }

        /** Get vertical baseline */
        double verticalBaseline() const { return _bv; }

        /** Get perpendicular baseline */
        double perpendicularBaseline() const { return (-1. * _bh * _coslook) + 
                                              (_bv * _sinlook); }

        /** Get sin of look angle */
        double sinLook() const { return _sinlook;}

        /** Get cos of look angle */
        double cosLook() const { return _coslook;}

        /** Reference ECEF position for baseline */
        cartesian_t refXyz() const { return _refxyz; }

        /** Unit vector in look direction */
        cartesian_t look() const { return _look; }

        /** Unit vector in radial direction */
        cartesian_t rhat() const { return _rhat; }

        /** Unit vector in cross track direction */
        cartesian_t chat() const { return _chat; }

        /** Unit vector in direction of velocity */
        cartesian_t vhat() const { return _vhat; }

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
        
        void init();
        void initBasis(double);
        cartesian_t calculateBasisOffset(const cartesian_t &) const;
        void computeBaselines();
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
        cartesian_t _refxyz, _look, _rhat, _chat, _vhat;
        // Baseline scalars
        double _bh, _bv;
        // Look angle
        double _sinlook, _coslook;
        // Velocity magnitude
        double _velocityMagnitude;
};

isce::core::Baseline & isce::core::Baseline::
operator=(const Baseline & rhs) {
    _orbit1 = rhs.orbit1();
    _orbit2 = rhs.orbit2();
    _radar = rhs.radar();
    _orbitMethod = rhs.orbitMethod();
    _refxyz = rhs.refXyz();
    _look = rhs.look();
    _rhat = rhs.rhat();
    _chat = rhs.chat();
    _vhat = rhs.vhat();
    _bh = rhs.horizontalBaseline();
    _bv = rhs.verticalBaseline();
    _sinlook = rhs.sinLook();
    _coslook = rhs.cosLook();
    _velocityMagnitude = rhs.velocityMagnitude();
    return *this;
}

#endif

// end of file
