//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "isce/core/Constants.h"
#include "isce/core/Ellipsoid.h"
#include "isce/core/LinAlg.h"
using isce::core::Ellipsoid;
using isce::core::LinAlg;
using isce::core::latLonConvMethod;
using std::invalid_argument;
using std::string;
using std::to_string;
using std::vector;

void Ellipsoid::latLon(vector<double> &v, vector<double> &llh, int ctype) {
    /* 
     * Given a conversion type ('ctype'), either converts a vector to lat, lon, and height
     * above the reference ellipsoid, or given a lat, lon, and height produces a geocentric
     * vector.
     */

    // Error checking to make sure inputs have expected characteristics
    checkVecLen(v,3);
    checkVecLen(llh,3);

    if (ctype == LLH_2_XYZ) {
        auto re = rEast(llh[0]);
        v[0] = (re + llh[2]) * cos(llh[0]) * cos(llh[1]);
        v[1] = (re + llh[2]) * cos(llh[0]) * sin(llh[1]);
        v[2] = ((re * (1. - e2)) + llh[2]) * sin(llh[0]);
    } else if (ctype == XYZ_2_LLH) {  // Translated from prior Python implementation (isceobj.Ellipsoid.xyz_to_llh)
        auto p = (pow(v[0], 2) + pow(v[1], 2)) / pow(a, 2);
        auto q = ((1. - e2) * pow(v[2], 2)) / pow(a, 2);
        auto r = (p + q - pow(e2, 2)) / 6.;
        auto s = (pow(e2, 2) * p * q) / (4. * pow(r, 3.));
        auto t = pow(1. + s + sqrt(s * (2. + s)), (1./3.));
        auto u = r * (1. + t + (1. / t));
        auto rv = sqrt(pow(u, 2) + (pow(e2, 2) * q));
        auto w = (e2 * (u + rv - q)) / (2. * rv);
        auto k = sqrt(u + rv + pow(w, 2)) - w;
        auto d = (k * sqrt(pow(v[0], 2) + pow(v[1], 2))) / (k + e2);
        llh[0] = atan2(v[2], d);
        llh[1] = atan2(v[1], v[0]);
        llh[2] = ((k + e2 - 1.) * sqrt(pow(d, 2) + pow(v[2], 2))) / k;
    } else if (ctype == XYZ_2_LLH_OLD) {  // Translated from prior Fortran implementation
        auto b = a * sqrt(1. - e2);
        auto p = sqrt(pow(v[0], 2) + pow(v[1], 2));
        auto tant = (v[2] / p) * sqrt(1. / (1. - e2));
        auto theta = atan(tant);
        tant = (v[2] + (((1. / (1. - e2)) - 1.) * b * pow(sin(theta), 3.))) / (p - (e2 * a * pow(cos(theta), 3.)));
        llh[0] = atan(tant);
        llh[1] = atan2(v[1], v[0]);
        llh[2] = (p / cos(llh[0])) - rEast(llh[0]);
    } else {
        string errstr = "Unrecognized conversion type in Ellipsoid::latLon. Expected one of:\n";
        errstr += "  LLH_2_XYZ (== " + to_string(LLH_2_XYZ) + ")\n";
        errstr += "  XYZ_2_LLH (== " + to_string(XYZ_2_LLH) + ")\n";
        errstr += "  XYZ_2_LLH_OLD (== " + to_string(XYZ_2_LLH_OLD) + ")\n";
        errstr += "Encountered conversion type " + to_string(ctype);
        throw invalid_argument(errstr);
    }
}

void Ellipsoid::getAngs(vector<double> &pos, vector<double> &vel, vector<double> &vec, double &az, double &lk) {
    /*
     * Computes the look vector given the look angle, azimuth angle, and position vector
     */
    
    // Error checking to make sure inputs have expected characteristics
    checkVecLen(pos,3);
    checkVecLen(vel,3);
    checkVecLen(vec,3);

    vector<double> temp(3);
    latLon(pos, temp, XYZ_2_LLH);
    
    vector<double> n = {-cos(temp[0])*cos(temp[1]), -cos(temp[0])*sin(temp[1]), -sin(temp[0])};
    lk = acos(LinAlg::dot(n, vec) / LinAlg::norm(vec));
    LinAlg::cross(n, vel, temp);
    
    vector<double> c(3);
    LinAlg::unitVec(temp, c);
    LinAlg::cross(c, n, temp);
    
    vector<double> t(3);
    LinAlg::unitVec(temp, t);
    az = atan2(LinAlg::dot(c, vec), LinAlg::dot(t, vec));
}

void Ellipsoid::getTCN_TCvec(vector<double> &pos, vector<double> &vel, vector<double> &vec, vector<double> &TCVec) {
    /*
     * Computes the projection of an xyz vector on the TC plane in xyz
     */
    
    // Error checking to make sure inputs have expected characteristics
    checkVecLen(pos,3);
    checkVecLen(vel,3);
    checkVecLen(vec,3);
    checkVecLen(TCVec,3);

    vector<double> temp(3);
    latLon(pos, temp, XYZ_2_LLH);
    
    vector<double> n = {-cos(temp[0])*cos(temp[1]), -cos(temp[0])*sin(temp[1]), -sin(temp[0])};
    LinAlg::cross(n, vel, temp);
    
    vector<double> c(3);
    LinAlg::unitVec(temp, c);
    LinAlg::cross(c, n, temp);
    
    vector<double> t(3);
    LinAlg::unitVec(temp, t);
    for (int i=0; i<3; i++) TCVec[i] = (LinAlg::dot(t, vec) * t[i]) + (LinAlg::dot(c, vec) * c[i]);
}

void Ellipsoid::TCNbasis(vector<double> &pos, vector<double> &vel, vector<double> &t, vector<double> &c, vector<double> &n) {
    
    // Error checking to make sure inputs have expected characteristics
    checkVecLen(pos,3);
    checkVecLen(vel,3);
    checkVecLen(t,3);
    checkVecLen(c,3);
    checkVecLen(n,3);

    vector<double> llh(3);
    latLon(pos,llh,XYZ_2_LLH);

    n = {-cos(llh[0]) * cos(llh[1]), -cos(llh[0]) * sin(llh[1]), -sin(llh[0])};
    
    vector<double> temp(3);
    LinAlg::cross(n,vel,temp);
    LinAlg::unitVec(temp,c);
    LinAlg::cross(c,n,temp);
    LinAlg::unitVec(temp,t);
}

