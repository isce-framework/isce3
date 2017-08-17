//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_GPULINALG_H
#define ISCELIB_GPULINALG_H

#include <cuda_runtime.h>

namespace isceLib {
    struct gpuLinAlg {
        gpuLinAlg() = default;

        __device__ static void cross(double*,double*,double*);
        __device__ static double dot(double*,double*);
        __device__ static void linComb(double,double*,double,double*,double*);
        __device__ static void unitVec(double*,double*);
    }
}

#endif

