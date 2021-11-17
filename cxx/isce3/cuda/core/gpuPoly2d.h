#pragma once

#include <isce3/core/forward.h>
#include <isce3/core/Common.h>

#include <cmath>

namespace isce3 { namespace cuda { namespace core {
    struct gpuPoly2d{
        int xOrder;
        int yOrder;
        double xMean;
        double yMean;
        double xNorm;
        double yNorm;
        double *coeffs;
        // True if copy-constructed from Orbit (on host),
        // False if copy-constructed from gpuOrbit (on device)
        bool owner;

        // Shallow-copy copy constructor only allowed on device, not host, but not allowed to free
        // own memory (host copy of gpuPoly2d is only one allowed)
        CUDA_DEV gpuPoly2d(int xo, int yo, double xm, double ym, double xn, double yn) : xOrder(xo),
                                                                                             yOrder(yo),
                                                                                             xMean(xm),
                                                                                             yMean(ym),
                                                                                             xNorm(xn),
                                                                                             yNorm(yn),
                                                                                             owner(false)
                                                                                             {}
        CUDA_HOSTDEV gpuPoly2d() : gpuPoly2d(-1,-1,0.,0.,1.,1.) {}
        // Advanced "copy constructor only allowed on host (manages deep copies from host to device)
        CUDA_DEV gpuPoly2d(const gpuPoly2d &p) : xOrder(p.xOrder), yOrder(p.yOrder),
                                                xMean(p.xMean), yMean(p.yMean),
                                                xNorm(p.xNorm), yNorm(p.yNorm),
                                                coeffs(p.coeffs), owner(false) {}
        CUDA_HOST gpuPoly2d(const isce3::core::Poly2d&);
        ~gpuPoly2d();

        CUDA_HOSTDEV inline gpuPoly2d& operator=(const gpuPoly2d&);

        CUDA_DEV double eval(double, double) const;

        // Host function to test underlying device function in a single-threaded context
        CUDA_HOST double eval_h(double, double);
    };


    CUDA_HOSTDEV inline gpuPoly2d& gpuPoly2d::operator=(const gpuPoly2d &rhs)
    {
        xOrder = rhs.xOrder;
        yOrder = rhs.yOrder;
        xMean = rhs.xMean;
        yMean = rhs.yMean;
        xNorm = rhs.xNorm;
        yNorm = rhs.yNorm;
        coeffs = rhs.coeffs;
        return *this;
    }


}}}
