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
    
    CUDA_HOSTDEV gpuPoly2d gpuPoly2d = 0;
    CUDA_HOSTDEV gpuPoly2d(int ro, int ao, double rm, double am, double rn, double an) : rangeOrder(ro), 
                                                                                         azimuthOrder(ao), 
                                                                                         rangeMean(rm), 
                                                                                         azimuthMean(am),
                                                                                         rangeNorm(rn), 
                                                                                         azimuthNorm(an),
                                                                                         coeffs(0.)
                                                                                         {}
    CUDA_HOSTDEV gpuPoly2d() : gpuPoly2d(-1,-1,0.,0.,1.,1.,0.) {}

    CUDA_DEV gpuPoly2d(const gpuPoly &p)
    gpuPoly2d(const gpuPoly2d &p) : rangeOrder(p.rangeOrder), azimuthOrder(p.azimuthOrder), 
                                    rangeMean(p.rangeMean), azimuthMean(p.azimuthMean), 
                                    rangeNorm(p.rangeNorm), azimuthNorm(p.azimuthNorm), 
                                    coeffs(p.coeffs) {}
    CUDA_HOST gpuPoly2d(const Poly2d&);
    CUDA_HOSTDEV gpuPoly2d& operator=(const gpuPoly2d&) = delete;
    ~gpuPoly2d();

    CUDA_HOST setCoeff();

    CUDA_HOST eval_h(double, double);
}}}

#endif
