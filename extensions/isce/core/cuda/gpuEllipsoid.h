//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_GPUELLIPSOID_H
#define ISCELIB_GPUELLIPSOID_H

#include <cmath>
#include "Ellipsoid.h"

namespace isceLib {
    struct gpuEllipsoid {
        double a;
        double e2;

        gpuEllipsoid(double _a, double _e2) : a(_a), e2(_e2) {}
        gpuEllipsoid() : gpuEllipsoid(0.,0.) {}
        gpuEllipsoid(const gpuEllipsoid &e) : a(e.a), e2(e.e2) {}
        gpuEllipsoid(const Ellipsoid &e) : a(e.a), e2(e.e2) {}

        __device__ inline double rEast(double);
        __device__ inline double rNorth(double);
        __device__ inline double rDir(double,double);
        __device__ void llh2xyz(double*,double*);
        __device__ void xyz2llh(double*,double*);
        __device__ void TCNbasis(double*,double*,double*,double*,double*);
    };

    __device__ inline double gpuEllipsoid::rEast(double lat) { return a / sqrt(1. - (e2 * pow(sin(lat), 2.))); }

    __device__ inline double gpuEllipsoid::rNorth(double lat) { return (a * (1. - e2)) / pow((1. - (e2 * pow(lat, 2.))), 1.5); }

    __device__ inline double gpuEllipsoid::rDir(double hdg, double lat) {
        double re = rEast(lat);
        double rn = rNorth(lat);
        return (re * rn) / ((re * pow(cos(hdg), 2.)) + (rn * pow(sin(hdg), 2.)));
    }
}

