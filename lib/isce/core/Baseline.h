//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CORE_BASELINE_H
#define ISCE_CORE_BASELINE_H

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

// Baseline declaration
class isce::core::Baseline {

    public:
        // Constructors
        Baseline() {}
        Baseline(const Baseline &b) : 
            _orbit1(b.orbit1()), _orbit2(b.orbit2()), _radar(b.radar()), _elp(b.ellipsoid()),
            _orbitMethod(b.orbitMethod()), _refxyz(b.refXyz()), _look(b.look()),
            _rhat(b.rhat()), _chat(b.chat()), _vhat(b.vhat()), _bh(b.horizontalBaseline()),
            _bv(b.verticalBaseline()), _sinlook(b.sinLook()),
            _coslook(b.cosLook()), _velocityMagnitude(b.velocityMagnitude()) {}
        inline Baseline& operator=(const Baseline&);

        // Get baselines
        double horizontalBaseline() const { return _bh; }
        double verticalBaseline() const { return _bv; }
        double perpendicularBaseline() const { return (-1. * _bh * _coslook) + 
                                              (_bv * _sinlook); }

        // Look angles
        double sinLook() const { return _sinlook;}
        double cosLook() const { return _coslook;}

        // Basis vectors
        cartesian_t refXyz() const { return _refxyz; }
        cartesian_t look() const { return _look; }
        cartesian_t rhat() const { return _rhat; }
        cartesian_t chat() const { return _chat; }
        cartesian_t vhat() const { return _vhat; }

        // Orbit method
        orbitInterpMethod orbitMethod() const { return _orbitMethod; }
        // Orbits
        Orbit orbit1() const { return _orbit1; }
        Orbit orbit2() const { return _orbit2; }
        // Metadata
        Metadata radar() const { return _radar; }
        // Ellipsoid
        Ellipsoid ellipsoid() const { return _elp; }    
        // Velocity magnitude
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
