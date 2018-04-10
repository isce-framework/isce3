//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include "Constants.h"
#include "LinAlg.h"
#include "Position.h"
using isce::core::LinAlg;
using isce::core::Position;

void Position::lookVec(double look, double az, cartesian_t & v) const {
    /*
     * Computes the look vector given the look angle, azimuth angle, and position vector.
     */
    cartesian_t n;
    LinAlg::unitVec(j, n);

    for (int i=0; i<3; i++) n[i] = -n[i];
    cartesian_t temp;
    LinAlg::cross(n, jdot, temp);

    cartesian_t c;
    LinAlg::unitVec(temp, c);
    LinAlg::cross(c, n, temp);

    cartesian_t t;
    LinAlg::unitVec(temp, t);
    LinAlg::linComb(cos(az), t, sin(az), c, temp);

    cartesian_t w;
    LinAlg::linComb(cos(look), n, sin(look), temp, w);
    LinAlg::unitVec(w, v);
}

// end of file
