//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_POSITION_H
#define ISCELIB_POSITION_H

#include <vector>

namespace isceLib {
    struct Position {
        std::vector<double> j;
        std::vector<double> jdot;
        std::vector<double> jddt;

        Position() : j(3), jdot(3), jddt(3) {}
        Position(const Position &p) : j(p.j), jdot(p.jdot), jddt(p.jddt) {}
        inline Position& operator=(const Position&);

        void lookVec(double,double,std::vector<double>&);
    };

    inline Position& Position::operator=(const Position &rhs) {
        j = rhs.j;
        jdot = rhs.jdot;
        jddt = rhs.jddt;
        return *this;
    }
}

#endif
