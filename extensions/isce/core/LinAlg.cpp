//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "LinAlg.h"
#include "isceLibConstants.h"
using isceLib::LinAlg;
using std::invalid_argument;
using std::string;
using std::to_string;
using std::vector;


void LinAlg::cross(vector<double> &u, vector<double> &v, vector<double> &w) {
    
    // Error checking
    checkVecLen(u,3);
    checkVecLen(v,3);
    checkVecLen(w,3);

    w[0] = (u[1] * v[2]) - (u[2] * v[1]);
    w[1] = (u[2] * v[0]) - (u[0] * v[2]);
    w[2] = (u[0] * v[1]) - (u[1] * v[0]);
}

double LinAlg::dot(vector<double> &v, vector<double> &w) {

    // Error checking
    checkVecLen(v,3);
    checkVecLen(w,3);

    return (v[0] * w[0]) + (v[1] * w[1]) + (v[2] * w[2]);
}

void LinAlg::linComb(double k1, vector<double> &u, double k2, vector<double> &v, vector<double> &w) {
    
    // Error checking
    checkVecLen(u,3);
    checkVecLen(v,3);
    checkVecLen(w,3);

    for (int i=0; i<3; i++) w[i] = (k1 * u[i]) + (k2 * v[i]);
}

void LinAlg::matMat(vector<vector<double>> &a, vector<vector<double>> &b, vector<vector<double>> &c) {
    
    // Error checking (not using checkVecLen since it's only 1D)
    check2dVecLen(a,3,3);
    check2dVecLen(b,3,3);
    check2dVecLen(c,3,3);

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            c[i][j] = (a[i][0] * b[0][j]) + (a[i][1] * b[1][j]) + (a[i][2] * b[2][j]);
        }
    }
}

void LinAlg::matVec(vector<vector<double>> &t, vector<double> &v, vector<double> &w) {
    
    // Error checking
    check2dVecLen(t,3,3);
    checkVecLen(v,3);
    checkVecLen(w,3);

    for (int i=0; i<3; i++) w[i] = (t[i][0] * v[0]) + (t[i][1] * v[1]) + (t[i][2] * v[2]);
}

double LinAlg::norm(vector<double> &v) {
    
    // Error checking
    checkVecLen(v,3);

    return sqrt(pow(v[0], 2.) + pow(v[1], 2.) + pow(v[2], 2.));
}

void LinAlg::tranMat(vector<vector<double>> &a, vector<vector<double>> &b) {

    // Error checking
    check2dVecLen(a,3,3);
    check2dVecLen(b,3,3);

    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            b[i][j] = a[j][i];
        }
    }
}

void LinAlg::unitVec(vector<double> &u, vector<double> &v) {

    // Error checking
    checkVecLen(u,3);
    checkVecLen(v,3);

    auto n = norm(u);
    if (n != 0.) {
        for (int i=0; i<3; i++) v[i] = u[i] / n;
    }
}

void LinAlg::enuBasis(double lat, double lon, vector<vector<double>> &enumat) {
    enumat = {-sin(lon), -sin(lat)*cos(lon), cos(lat)*cos(lon),
              cos(lon),  -sin(lat)*sin(lon), cos(lat)*sin(lon),
              0.,        cos(lat),           sin(lat)         };
}

