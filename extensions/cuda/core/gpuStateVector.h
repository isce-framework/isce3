//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#ifndef ISCE_CUDA_CORE_GPUSTATEVECTOR_H
#define ISCE_CUDA_CORE_GPUSTATEVECTOR_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#endif

// isce::core
#include "isce/core/StateVector.h"

// Declaration
namespace isce {
    namespace cuda {
        namespace core {
            class gpuStateVector;
        }
    }
}

// gpuStateVector definition
class isce::cuda::core::gpuStateVector {

    public:
        // Constructors
        CUDA_HOSTDEV inline gpuStateVector() {}
        CUDA_HOST inline gpuStateVector(const isce::core::StateVector & state) {
            position_h(state.position());
            velocity_h(state.velocity());
            time_h(state.date().secondsOfDay());
        }

        // Make data public to simplify access and to reduce use of temp variables
        // in device code
        double position[3];
        double velocity[3];
        double time;

        // Set position on host
        CUDA_HOST inline void position_h(const cartesian_t & p) {
            for (int i = 0; i < 3; ++i) {
                position[i] = p[i];
            }
        }

        // Set velocity on host
        CUDA_HOST inline void velocity_h(const cartesian_t & v) {
            for (int i = 0; i < 3; ++i) {
                velocity[i] = v[i];
            }
        }

        // Set epoch UTC time on host
        CUDA_HOST inline void time_h(double t) { time = t; }
};

#endif

// end of file
