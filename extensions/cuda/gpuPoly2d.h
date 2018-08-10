//
// Author: Liang Yu
// Copyright 2018
//

#ifndef __ISCE_CORE_CUDA_GPUPOLY2D_H__
#define __ISCE_CORE_CUDA_GPUPOLY2D_H__

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#endif

#include <cmath>
#include "Poly2d.h"

namespace isce { namespace core { namespace cuda {
    struct gpuPoly2d{
        int rangeOrder;
        int azimuthOrder;
        double rangeMean;
        double azimuthMean;
        double rangeNorm;
        double azimuthNorm;
        double *coeffs;
    }
    
    CUDA_HOSTDEV gpuPoly2d(int ro, int ao, double rm, double am, double rn, double an) : rangeOrder(ro), 
                                                                                         azimuthOrder(ao), 
                                                                                         rangeMean(rm), 
                                                                                         azimuthMean(am),
                                                                                         rangeNorm(rn), 
                                                                                         azimuthNorm(an),
                                                                                         coeffs(0.)
                                                                                         {}
    CUDA_HOSTDEV gpuPoly2d() : gpuPoly2d(-1,-1,0.,0.,1.,1.,0.) {}
}}}

#endif
