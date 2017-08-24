//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <vector>
#include "isce/core/Constants.h"
#include "isce/core/LinAlg.h"
#include "isce/core/Position.h"
using isce::core::LinAlg;
using isce::core::Position;
using std::vector;


void Position::lookVec(double look, double az, vector<double> &v) {
    /*
     * Computes the look vector given the look angle, azimuth angle, and position vector.
     */
    
    checkVecLen(v,3);

    vector<double> n(3);
    LinAlg::unitVec(j, n);
    
    for (int i=0; i<3; i++) n[i] = -n[i];
    vector<double> temp(3);
    LinAlg::cross(n, jdot, temp);

    vector<double> c(3);
    LinAlg::unitVec(temp, c);
    LinAlg::cross(c, n, temp);

    vector<double> t(3);
    LinAlg::unitVec(temp, t);
    LinAlg::linComb(cos(az), t, sin(az), c, temp);

    vector<double> w(3);
    LinAlg::linComb(cos(look), n, sin(look), temp, w);
    LinAlg::unitVec(w, v);
}

