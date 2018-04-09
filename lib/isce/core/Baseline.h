//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_BASELINE_H__
#define __ISCE_CORE_BASELINE_H__

#include <vector>
#include "Constants.h"
#include "Ellipsoid.h"
#include "Metadata.h"
#include "Orbit.h"

namespace isce { namespace core {
    struct Baseline {
        Orbit orbit1, orbit2;
        RadarMetadata radar;
        Ellipsoid elp;
        orbitInterpMethod orbit_method;
        // Basis vectors
        cartesian_t refxyz, look, rhat, chat, vhat;
        // Baseline scalars
        double bh, bv;
        // Look vector components
        double sinlook, coslook;
        // Velocity magnitude
        double velocityMagnitude;

        Baseline() {}
        Baseline(const Baseline &b) : orbit1(b.orbit1), orbit2(b.orbit2), radar(b.radar),
                                      elp(b.elp), orbit_method(b.orbit_method), refxyz(b.refxyz),
                                      look(b.look), rhat(b.rhat), chat(b.chat), vhat(b.vhat),
                                      bh(b.bh), bv(b.bv), sinlook(b.sinlook), coslook(b.coslook),
                                      velocityMagnitude(b.velocityMagnitude) {}
        inline Baseline& operator=(const Baseline&);

        inline double getHorizontalBaseline() const { return bh; }
        inline double getVerticalBaseline() const { return bv; }
        inline double getPerpendicularBaseline() const { return (-1. * bh * coslook) + 
                                                                (bv * sinlook); }
        void init();
        void initBasis(double);
        cartesian_t calculateBasisOffset(const cartesian_t &) const;
        void computeBaselines();
        void calculateLookVector(double);
    };

    inline Baseline& Baseline::operator=(const Baseline &rhs) {
        orbit1 = rhs.orbit1;
        orbit2 = rhs.orbit2;
        radar = rhs.radar;
        orbit_method = rhs.orbit_method;
        refxyz = rhs.refxyz;
        look = rhs.look;
        rhat = rhs.rhat;
        chat = rhs.chat;
        vhat = rhs.vhat;
        bh = rhs.bh;
        bv = rhs.bv;
        sinlook = rhs.sinlook;
        coslook = rhs.coslook;
        velocityMagnitude = rhs.velocityMagnitude;
        return *this;
    }
}}

#endif
