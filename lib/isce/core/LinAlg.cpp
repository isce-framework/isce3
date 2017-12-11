//
// Author: Joshua Cohen
// Copyright 2017
//
// Note: This class may be deprecated in the future given the existence of production linear algebra
// libraries

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "Constants.h"
#include "LinAlg.h"
using isce::core::LinAlg;
using std::invalid_argument;
using std::string;
using std::to_string;
using std::vector;


void LinAlg::cross(const vector<double> &u, const vector<double> &v, vector<double> &w) {
    /*
     *  Calculate the vector cross product of two 1x3 vectors (u, v) and store the resulting vector
     *  in w.
     */

    // Error checking
    checkVecLen(u,3);
    checkVecLen(v,3);
    checkVecLen(w,3);

    w[0] = (u[1] * v[2]) - (u[2] * v[1]);
    w[1] = (u[2] * v[0]) - (u[0] * v[2]);
    w[2] = (u[0] * v[1]) - (u[1] * v[0]);
}

double LinAlg::dot(const vector<double> &v, const vector<double> &w) {
    /*
     *  Calculate the vector dot product of two 1x3 vectors and return the result.
     */

    // Error checking
    checkVecLen(v,3);
    checkVecLen(w,3);

    return (v[0] * w[0]) + (v[1] * w[1]) + (v[2] * w[2]);
}

void LinAlg::linComb(double k1, const vector<double> &u, double k2, const vector<double> &v,
                     vector<double> &w) {
    /*
     *  Calculate the linear combination of two pairs of scalars and 1x3 vectors and store the
     *  resulting vector in w.
     */

    // Error checking
    checkVecLen(u,3);
    checkVecLen(v,3);
    checkVecLen(w,3);

    for (int i=0; i<3; i++) w[i] = (k1 * u[i]) + (k2 * v[i]);
}

void LinAlg::matMat(const vector<vector<double>> &a, const vector<vector<double>> &b,
                    vector<vector<double>> &c) {
    /*
     *  Calculate the matrix product of two 3x3 matrices and store the resulting matrix in c.
     */

    // Error checking
    check2dVecLen(a,3,3);
    check2dVecLen(b,3,3);
    check2dVecLen(c,3,3);

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            c[i][j] = (a[i][0] * b[0][j]) + (a[i][1] * b[1][j]) + (a[i][2] * b[2][j]);
        }
    }
}

void LinAlg::matVec(const vector<vector<double>> &t, const vector<double> &v, vector<double> &w) {
    /*
     *  Calculate the matrix product of a 1x3 vector with a 3x3 matrix and store the resulting
     *  vector in w.
     */

    // Error checking
    check2dVecLen(t,3,3);
    checkVecLen(v,3);
    checkVecLen(w,3);

    for (int i=0; i<3; i++) w[i] = (t[i][0] * v[0]) + (t[i][1] * v[1]) + (t[i][2] * v[2]);
}

double LinAlg::norm(const vector<double> &v) {
    /*
     *  Calculate the magnitude of a 1x3 vector and return the result
     */

    // Error checking
    checkVecLen(v,3);

    return sqrt(pow(v[0], 2.) + pow(v[1], 2.) + pow(v[2], 2.));
}

void LinAlg::tranMat(const vector<vector<double>> &a, vector<vector<double>> &b) {
    /*
     *  Transpose a 3x3 matrix and store the resulting matrix in b.
     */


    // Error checking
    check2dVecLen(a,3,3);
    check2dVecLen(b,3,3);

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            b[i][j] = a[j][i];
        }
    }
}

void LinAlg::unitVec(const vector<double> &u, vector<double> &v) {
    /*
     *  Calculate the normalized unit vector from a 1x3 vector and store the resulting vector in v.
     */

    // Error checking
    checkVecLen(u,3);
    checkVecLen(v,3);

    auto n = norm(u);
    if (n != 0.) {
        for (int i=0; i<3; i++) v[i] = u[i] / n;
    }
}

void LinAlg::enuBasis(double lat, double lon, vector<vector<double>> &enumat) {
    /*
     *
     */

    // Error checking
    check2dVecLen(enumat,3,3);

    enumat = {{-sin(lon), -sin(lat)*cos(lon), cos(lat)*cos(lon)},
              {cos(lon),  -sin(lat)*sin(lon), cos(lat)*sin(lon)},
              {0.,        cos(lat),           sin(lat)         }};
}
