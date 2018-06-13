//
// Author: Joshua Cohen
// Copyright 2017
//
// Note: This class may be deprecated in the future given the existence of production linear algebra
// libraries

#include <cmath>
#include <stdexcept>
#include <vector>
#include "Constants.h"
#include "LinAlg.h"

void isce::core::LinAlg::
scale(cartesian_t & v, double scaleFactor) {
    /*
     * Scale all elements by a scale factor.
     */
    for (int i = 0; i < 3; ++i) v[i] *= scaleFactor;
}

void isce::core::LinAlg::
cross(const cartesian_t & u, const cartesian_t & v, cartesian_t & w) {
    /*
     *  Calculate the vector cross product of two 1x3 vectors (u, v) and store the resulting vector
     *  in w.
     */
    w[0] = (u[1] * v[2]) - (u[2] * v[1]);
    w[1] = (u[2] * v[0]) - (u[0] * v[2]);
    w[2] = (u[0] * v[1]) - (u[1] * v[0]);
}

double isce::core::LinAlg::
dot(const cartesian_t & v, const cartesian_t & w) {
    /*
     *  Calculate the vector dot product of two 1x3 vectors and return the result.
     */
    return (v[0] * w[0]) + (v[1] * w[1]) + (v[2] * w[2]);
}

void isce::core::LinAlg::
linComb(double k1, const cartesian_t & u, double k2, const cartesian_t & v,
        cartesian_t & w) {
    /*
     *  Calculate the linear combination of two pairs of scalars and 1x3 vectors and store the
     *  resulting vector in w.
     */
    for (int i = 0; i < 3; ++i) w[i] = (k1 * u[i]) + (k2 * v[i]);
}

void isce::core::LinAlg::
matMat(const cartmat_t & a, const cartmat_t & b, cartmat_t & c) {
    /*
     *  Calculate the matrix product of two 3x3 matrices and store the resulting matrix in c.
     */
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; ++j) {
            c[i][j] = (a[i][0] * b[0][j]) + (a[i][1] * b[1][j]) + (a[i][2] * b[2][j]);
        }
    }
}

void isce::core::LinAlg::
matVec(const cartmat_t & t, const cartesian_t & v, cartesian_t & w) {
    /*
     *  Calculate the matrix product of a 1x3 vector with a 3x3 matrix and store the resulting
     *  vector in w.
     */
    for (int i=0; i<3; ++i) w[i] = (t[i][0] * v[0]) + (t[i][1] * v[1]) + (t[i][2] * v[2]);
}

double isce::core::LinAlg::
norm(const cartesian_t & v) {
    /*
     *  Calculate the magnitude of a 1x3 vector and return the result
     */
    return std::sqrt(std::pow(v[0], 2) + std::pow(v[1], 2) + std::pow(v[2], 2));
}

void isce::core::LinAlg::
tranMat(const cartmat_t & a, cartmat_t & b) {
    /*
     *  Transpose a 3x3 matrix and store the resulting matrix in b.
     */
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            b[i][j] = a[j][i];
        }
    }
}

void isce::core::LinAlg::
unitVec(const cartesian_t & u, cartesian_t & v) {
    /*
     *  Calculate the normalized unit vector from a 1x3 vector and store the resulting vector in v.
     */
    auto n = norm(u);
    if (n != 0.0) {
        for (int i = 0; i < 3; ++i) {
            v[i] = u[i] / n;
        }
    }
}


/** @param[in] lat Latitude in radians
 *  @param[in] lon Longitude in radians
 *  @param[out] enumat Matrix with rotation matrix*/
void isce::core::LinAlg::
enuBasis(double lat, double lon, cartmat_t & enumat) {
    /*
     *
     */
    enumat = {{{-std::sin(lon), -std::sin(lat)*std::cos(lon), std::cos(lat)*std::cos(lon)},
               {std::cos(lon),  -std::sin(lat)*std::sin(lon), std::cos(lat)*std::sin(lon)},
               {0.0, std::cos(lat), std::sin(lat)}}};
}

// end of file
