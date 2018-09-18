//
// Author: Liang Yu
// Copyright 2018
//
// NOTE: gpuOrbit used as template

#include <cuda_runtime.h>
#include "gpuInterpolator.h"

using isce::cuda::core::gpuInterpolator;


template <class U>
__device__ U isce::cuda::core::gpuBilinearInterpolator<U>::interpolate(double x, double y, const U* z, size_t nx) {

    size_t x1 = floor(x);
    size_t x2 = ceil(x);
    size_t y1 = floor(y);
    size_t y2 = ceil(y);
    U q11 = z[y1*nx + x1];
    U q12 = z[y2*nx + x1];
    U q21 = z[y1*nx + x2];
    U q22 = z[y2*nx + x2];

    if ((y1 == y2) && (x1 == x2)) {
        return q11;
    } else if (y1 == y2) {
        return ((U)((x2 - x) / (x2 - x1)) * q11) +
               ((U)((x - x1) / (x2 - x1)) * q21);
    } else if (x1 == x2) {
        return ((U)((y2 - y) / (y2 - y1)) * q11) +
               ((U)((y - y1) / (y2 - y1)) * q12);
    } else {
        return  ((q11 * (U)((x2 - x) * (y2 - y))) /
                 (U)((x2 - x1) * (y2 - y1))) +
                ((q21 * (U)((x - x1) * (y2 - y))) /
                 (U)((x2 - x1) * (y2 - y1))) +
                ((q12 * (U)((x2 - x) * (y - y1))) /
                 (U)((x2 - x1) * (y2 - y1))) +
                ((q22 * (U)((x - x1) * (y - y1))) /
                 (U)((x2 - x1) * (y2 - y1)));
    }
}


