//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_POSITION_H__
#define __ISCE_CORE_POSITION_H__

#include <vector>
#include "Constants.h"

namespace isce { namespace core {
    struct Position {
        cartesian_t j;
        cartesian_t jdot;
        cartesian_t jddt;

        Position() {}
        Position(const Position &p) : j(p.j), jdot(p.jdot), jddt(p.jddt) {}
        inline Position& operator=(const Position&);

        void lookVec(double, double, cartesian_t&) const;
    };

    inline Position& Position::operator=(const Position &rhs) {
        j = rhs.j;
        jdot = rhs.jdot;
        jddt = rhs.jddt;
        return *this;
    }
}}

#endif
