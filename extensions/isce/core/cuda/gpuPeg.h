//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_GPUPEG_H
#define ISCELIB_GPUPEG_H

#include <cuda_runtime.h>
#include "Peg.h"

namespace isceLib {
    struct gpuPeg {
        double lat;
        double lon;
        double hdg;

        gpuPeg() = delete;
        __device__ gpuPeg(const gpuPeg &p) : lat(p.lat), lon(p.lon), hdg(p.hdg) {}
        gpuPeg(const Peg &p) : lat(p.lat), lon(p.lon), hdg(p.hdg) {}
        gpuPeg& operator=(const gpuPeg&) = delete;
    }
}

#endif
