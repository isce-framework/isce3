//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_POSITION_H__
#define __ISCE_CORE_POSITION_H_

#include <vector>

namespace isce { namespace core {
    struct Position {
        std::vector<double> j;
        std::vector<double> jdot;
        std::vector<double> jddt;

        Position() : j(3), jdot(3), jddt(3) {}                              // Default constructor
        Position(const Position &p) : j(p.j), jdot(p.jdot), jddt(p.jddt) {} // Copy constructor
        inline Position& operator=(const Position&);

        void lookVec(double,double,std::vector<double>&);
    };

    inline Position& Position::operator=(const Position &rhs) {
        j = rhs.j;
        jdot = rhs.jdot;
        jddt = rhs.jddt;
        return *this;
    }
}}

#endif
