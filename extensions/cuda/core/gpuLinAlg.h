//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCE_CUDA_CORE_GPULINALG_H
#define ISCE_CUDA_CORE_GPULINALG_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#endif

namespace isce { namespace cuda { namespace core {
    struct gpuLinAlg {
        CUDA_HOSTDEV gpuLinAlg() = delete;

        CUDA_DEV static void cross(double*,double*,double*);
        CUDA_DEV static double dot(double*,double*);
        CUDA_DEV static void linComb(double,double*,double,double*,double*);
        CUDA_DEV static void unitVec(double*,double*);
        CUDA_DEV static double norm(double*);
    };
}}}

#endif

// end of file
