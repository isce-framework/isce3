//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_CUDA_GPUPEG_H__
#define __ISCE_CORE_CUDA_GPUPEG_H__

#include "Peg.h"

namespace isce { namespace core { namespace cuda {
    struct gpuPeg {
        double lat;
        double lon;
        double hdg;

        __host__ __device__ gpuPeg(double _lat, double _lon, double _hdg) : lat(_lat), lon(_lon), 
                                                                            hdg(_hdg) {}
        __host__ __device__ gpuPeg() : gpuPeg(0.,0.,0.) {}
        __host__ __device__ gpuPeg(const gpuPeg &p) : lat(p.lat), lon(p.lon), hdg(p.hdg) {}
        // Alternate "copy" constructor from Peg object
        __host__ __device__ gpuPeg(const Peg &p) : lat(p.lat), lon(p.lon), hdg(p.hdg) {}
        __host__ __device__ inline gpuPeg& operator=(const gpuPeg&);
    };

    __host__ __device__ inline gpuPeg& gpuPeg::operator=(const gpuPeg &rhs) {
        lat = rhs.lat;
        lon = rhs.lon;
        hdg = rhs.hdg;
        return *this;
    }
}}}

#endif
