//
// Author: Liang Yu
// Copyright 2018
//

#ifndef __ISCE_CUDA_CORE_GPUPOLY2D_H__
#define __ISCE_CUDA_CORE_GPUPOLY2D_H__

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

using isce::core::Poly2d;

namespace isce { namespace cuda { namespace core {
    struct gpuPoly2d{
        int rangeOrder;
        int azimuthOrder;
        double rangeMean;
        double azimuthMean;
        double rangeNorm;
        double azimuthNorm;
        double *coeffs;
        // True if copy-constructed from Orbit (on host), 
        // False if copy-constructed from gpuOrbit (on device)
        bool owner;
    
        // Shallow-copy copy constructor only allowed on device, not host, but not allowed to free 
        // own memory (host copy of gpuPoly2d is only one allowed)
        CUDA_DEV gpuPoly2d(int ro, int ao, double rm, double am, double rn, double an) : rangeOrder(ro), 
                                                                                             azimuthOrder(ao), 
                                                                                             rangeMean(rm), 
                                                                                             azimuthMean(am),
                                                                                             rangeNorm(rn), 
                                                                                             azimuthNorm(an), 
                                                                                             owner(false)
                                                                                             {}
        CUDA_HOSTDEV gpuPoly2d() : gpuPoly2d(-1,-1,0.,0.,1.,1.) {}
        // Advanced "copy constructor only allowed on host (manages deep copies from host to device)
        CUDA_DEV gpuPoly2d(const gpuPoly2d &p) : rangeOrder(p.rangeOrder), azimuthOrder(p.azimuthOrder), 
                                                rangeMean(p.rangeMean), azimuthMean(p.azimuthMean), 
                                                rangeNorm(p.rangeNorm), azimuthNorm(p.azimuthNorm), 
                                                coeffs(p.coeffs), owner(false) {}
        CUDA_HOST gpuPoly2d(const Poly2d&);
        ~gpuPoly2d();

        CUDA_HOSTDEV inline gpuPoly2d& operator=(const gpuPoly2d&);

        CUDA_DEV void eval(double, double, double*);

        // Host function to test underlying device function in a single-threaded context
        CUDA_HOST double eval_h(double, double); 
    };


    CUDA_HOSTDEV inline gpuPoly2d& gpuPoly2d::operator=(const gpuPoly2d &rhs)
    {
        rangeOrder = rhs.rangeOrder;
        azimuthOrder = rhs.azimuthOrder;
        rangeMean = rhs.rangeMean;
        azimuthMean = rhs.azimuthMean;
        rangeNorm = rhs.rangeNorm;
        azimuthNorm = rhs.azimuthNorm;
        coeffs = rhs.coeffs;
        return *this;
    }


}}}

#endif
