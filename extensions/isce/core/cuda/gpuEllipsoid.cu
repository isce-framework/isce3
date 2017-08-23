//
// Author: Joshua Cohen
// Copyright 2017
//

#include <cmath>
#include "gpuEllipsoid.h"
#include "gpuLinAlg.h"
using isceLib::gpuEllipsoid;
using isceLib::gpuLinAlg;

__device__ void gpuEllipsoid::llh2xyz(double *xyz, double *llh) {
    double re = rEast(llh[0]);
    xyz[0] = (re + llh[2]) * cos(llh[0]) * cos(llh[1]);
    xyz[1] = (re + llh[2]) * cos(llh[0]) * sin(llh[1]);
    xyz[2] = ((re * (1. - e2)) + llh[2]) * sin(llh[0]);
}

__device__ void gpuEllipsoid::xyz2llh(double *xyz, double *llh) {
    double p = (pow(xyz[0],2) + pow(xyz[1],2)) / pow(a,2);
    double q = ((1. - e2) * pow(xyz[2],2)) / pow(a,2);
    double r = (p + q - pow(e2,2)) / 6.;
    double s = (pow(e2,2) * p * q) / (4. * pow(r,3));
    double t = cbrt(1. + s + sqrt(s * (2. + s)));
    double u = r * (1. + t + (1. / t));
    double rv = sqrt(pow(u,2) + (pow(e2,2) * q));
    double w = (e2 * (u + rv - q)) / (2. * rv);
    double k = sqrt(u + rv + pow(w,2)) - w;
    double d = (k * sqrt(pow(xyz[0],2) + pow(xyz[1],2))) / (k + e2);
    llh[0] = atan2(xyz[2],d);
    llh[1] = atan2(xyz[1],xyz[0]);
    llh[2] = ((k + e2 - 1.) * sqrt(pow(d,2) + pow(xyz[2],2))) / k;
}

__device__ void gpuEllipsoid::TCNbasis(double *pos, double *vel, double *t, double *c, double *n) {
    double temp[3];
    xyz2llh(pos,temp);
    n[0] = -cos(temp[0]) * cos(temp[1]);
    n[1] = -cos(temp[0]) * sin(temp[1]);
    n[2] = -sin(temp[0]);
    gpuLinAlg::cross(n,vel,temp);
    gpuLinAlg::unitVec(temp,c);
    gpuLinAlg::cross(c,n,temp);
    gpuLinAlg::unitVec(temp,t);
}

