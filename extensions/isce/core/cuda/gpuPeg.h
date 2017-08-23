//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_GPUPEG_H
#define ISCELIB_GPUPEG_H

#include "Peg.h"

namespace isceLib {
    struct gpuPeg {
        double lat;
        double lon;
        double hdg;

        __host__ __device__ gpuPeg(double _lat, double _lon, double _hdg) : lat(_lat), lon(_lon), hdg(_hdg) {}  // Value constructor
        __host__ __device__ gpuPeg() : gpuPeg(0.,0.,0.) {}                                                      // Default constructor (delegated)
        __host__ __device__ gpuPeg(const gpuPeg &p) : lat(p.lat), lon(p.lon), hdg(p.hdg) {}                     // Copy constructor
        __host__ __device__ gpuPeg(const Peg &p) : lat(p.lat), lon(p.lon), hdg(p.hdg) {}                        // Alternate "copy" constructor from Peg object
        __host__ __device__ inline gpuPeg& operator=(const gpuPeg&);
    };

    __host__ __device__ inline gpuPeg& gpuPeg::operator=(const gpuPeg &rhs) {
        lat = rhs.lat;
        lon = rhs.lon;
        hdg = rhs.hdg;
        return *this;
    }
}

#endif
