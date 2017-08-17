//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_GPUPEGTRANS_H
#define ISCELIB_GPUPEGTRANS_H

#include <cuda_runtime.h>
#include "gpuEllipsoid.h"
#include "gpuPeg.h"

namespace isceLib {
    struct gpuPegtrans {
        double mat[3][3];
        double matinv[3][3];
        double ov[3];
        double radcur;

        gpuPegtrans() : radcur(0.) {}
        __device__ gpuPegtrans(const gpuPegtrans &p);
        gpuPegtrans& operator=(const gpuPegtrans&) = delete;

        __device__ void radar2xyz(gpuEllipsoid&,gpuPeg&);
        __device__ void xyz2sch(double*,double*);
    }
}

#endif
