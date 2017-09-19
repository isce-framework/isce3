//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_CUDA_GPUELLIPSOID_H__
#define __ISCE_CORE_CUDA_GPUELLIPSOID_H__

#include <cmath>
#include "../Ellipsoid.h"

namespace isce { namespace core { namespace cuda {
    struct gpuEllipsoid {
        double a;
        double e2;

        __host__ __device__ gpuEllipsoid(double maj, double ecc) : a(maj), e2(ecc) {}
        __host__ __device__ gpuEllipsoid() : gpuEllipsoid(0.,0.) {}
        __host__ __device__ gpuEllipsoid(const gpuEllipsoid &e) : a(e.a), e2(e.e2) {}
        // Alternate "copy" constructor from Ellipsoid object
        __host__ gpuEllipsoid(const Ellipsoid &e) : a(e.a), e2(e.e2) {}
        __host__ __device__ inline gpuEllipsoid& operator=(const gpuEllipsoid&);

        __device__ inline double rEast(double);
        __device__ inline double rNorth(double);
        __device__ inline double rDir(double,double);
        __device__ void llh2xyz(double*,double*);
        __device__ void xyz2llh(double*,double*);
        __device__ void TCNbasis(double*,double*,double*,double*,double*);
    };

    __host__ __device__ inline gpuEllipsoid& gpuEllipsoid::operator=(const gpuEllipsoid &rhs) {
        a = rhs.a;
        e2 = rhs.e2;
        return *this;
    }

    __device__ inline double gpuEllipsoid::rEast(double lat) { 
        return a / sqrt(1. - (e2 * pow(sin(lat), 2))); 
    }

    __device__ inline double gpuEllipsoid::rNorth(double lat) { 
        return (a * (1. - e2)) / pow((1. - (e2 * pow(lat, 2))), 1.5); 
    }

    __device__ inline double gpuEllipsoid::rDir(double hdg, double lat) {
        double re = rEast(lat);
        double rn = rNorth(lat);
        return (re * rn) / ((re * pow(cos(hdg), 2)) + (rn * pow(sin(hdg), 2)));
    }
}}}

#endif
