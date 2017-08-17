//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_GPUORBIT_H
#define ISCELIB_GPUORBIT_H

#include <cuda_runtime.h>
#include "Orbit.h"

namespace isceLib {
    struct gpuOrbit {
        int nVectors;
        double *UTCtime;
        double *position;
        double *velocity;
        bool owner; // True if copy-constructed from Orbit (on host), False if copy-constructed from gpuOrbit (on device)

        // Suppress standard constructors since there is undefined behavior between CPU/GPU implementations
        gpuOrbit() = delete;
        __device__ gpuOrbit(const gpuOrbit &o) : nVectors(o.nVectors), UTCtime(o.UTCtime), position(o.position), velocity(o.velocity), owner(false) {}
        gpuOrbit(const Orbit&);
        gpuOrbit& operator=(const gpuOrbit&) = delete;
        ~gpuOrbit();

        __device__ inline void getStateVector(int,double&,double*,double*);
        __device__ int interpolateWGS84Orbit(double,double*,double*);
        __device__ int interpolateLegendreOrbit(double,double*,double*);
        __device__ int interpolateSCHOrbit(double,double*,double*);
        __device__ int computeAcceleration(double,double*);
    };

    __device__ inline void gpuOrbit::getStateVector(int idx, double &t, double *pos, double *vel) {
        // Note we can't really do much in the way of bounds-checking since we can't use the <stdexcept> library
        if ((idx < 0) || (idx >= nVectors)) return;
        t = UTCtime[idx];
        for (int i=0; i<3; i++) {
            pos[i] = position[3*idx+i];
            vel[i] = velocity[3*idx+i];
        }
    }
}

#endif
