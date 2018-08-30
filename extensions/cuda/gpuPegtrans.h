//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CUDA_CORE_GPUPEGTRANS_H__
#define __ISCE_CUDA_CORE_GPUPEGTRANS_H__

#include "gpuEllipsoid.h"
#include "gpuPeg.h"

namespace isce { namespace cuda { namespace core {
    struct gpuPegtrans {
        double mat[3][3];
        double matinv[3][3];
        double ov[3];
        double radcur;

        __host__ __device__ gpuPegtrans(double rdc) : radcur(rdc) {}
        __host__ __device__ gpuPegtrans() : gpuPegtrans(0.) {}
        __host__ __device__ gpuPegtrans(const gpuPegtrans&) = delete;
        __host__ __device__ gpuPegtrans& operator=(const gpuPegtrans&) = delete;

        __device__ void radar2xyz(gpuEllipsoid&,gpuPeg&);
        __device__ void xyz2sch(double*,double*);
    };
}}}

#endif
