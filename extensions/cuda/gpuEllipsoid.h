//
// Author: Joshua Cohen, Liang Yu
// Copyright 2017-2018
//

#ifndef __ISCE_CUDA_CORE_GPUELLIPSOID_H__
#define __ISCE_CUDA_CORE_GPUELLIPSOID_H__

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#define CUDA_GLOBAL
#endif

#include <cmath>
#include "Constants.h"
#include "Ellipsoid.h"

using isce::core::Ellipsoid;
using isce::core::cartesian_t;

namespace isce { namespace cuda { namespace core {
    struct gpuEllipsoid {
        double a;
        double e2;

        CUDA_HOSTDEV gpuEllipsoid(double maj, double ecc) : a(maj), e2(ecc) {}
        CUDA_HOSTDEV gpuEllipsoid() : gpuEllipsoid(0.,0.) {}
        CUDA_HOSTDEV gpuEllipsoid(const gpuEllipsoid &e) : a(e.a), e2(e.e2) {}
        // Alternate "copy" constructor from Ellipsoid object
        CUDA_HOST gpuEllipsoid(const Ellipsoid &e) : a(e.a()), e2(e.e2()) {}
        CUDA_HOSTDEV inline gpuEllipsoid& operator=(const gpuEllipsoid&);

        CUDA_DEV inline double rEast(double) const;
        CUDA_DEV inline double rNorth(double) const;
        CUDA_DEV inline double rDir(double,double) const;
        CUDA_DEV void lonLatToXyz(const double*,double*) const;
        CUDA_DEV void xyzToLonLat(const double*,double*) const;
        CUDA_DEV void TCNbasis(double*,double*,double*,double*,double*) const;
        
        /** Return eccentricity^2 */
        CUDA_HOSTDEV double gete2() const {return e2;}
        /** Return semi-major axis */
        CUDA_HOSTDEV double geta() const {return a;}

        // Host functions to test underlying device functions in a single-threaded context
        CUDA_HOST void lonLatToXyz_h(cartesian_t&,cartesian_t&);
        CUDA_HOST void xyzToLonLat_h(cartesian_t&,cartesian_t&);
    };

    CUDA_HOSTDEV inline gpuEllipsoid& gpuEllipsoid::operator=(const gpuEllipsoid &rhs) {
        a = rhs.a;
        e2 = rhs.e2;
        return *this;
    }

    CUDA_DEV inline double gpuEllipsoid::rEast(double lat) const{ 
        return a / sqrt(1. - (e2 * pow(sin(lat), 2))); 
    }

    CUDA_DEV inline double gpuEllipsoid::rNorth(double lat) const { 
        return (a * (1. - e2)) / pow((1. - (e2 * pow(lat, 2))), 1.5); 
    }

    CUDA_DEV inline double gpuEllipsoid::rDir(double hdg, double lat) const{
        double re = rEast(lat);
        double rn = rNorth(lat);
        return (re * rn) / ((re * pow(cos(hdg), 2)) + (rn * pow(sin(hdg), 2)));
    }
}}}

#endif
