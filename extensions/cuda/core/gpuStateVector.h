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

        // Get position
        CUDA_DEV inline double * position() const {
            static double output[3];
            for (int i = 0; i < 3; ++i) {
                output[i] = _position[i];
            }
            return output;
        }

        // Set position on device
        CUDA_DEV inline void position(double * p) {
            for (int i = 0; i < 3; ++i) {
                _position[i] = p[i];
            }
        }

        // Set position on host
        CUDA_HOST inline void position_h(const cartesian_t & p) {
            for (int i = 0; i < 3; ++i) {
                _position[i] = p[i];
            }
        }

        // Get velocity
        CUDA_DEV inline double * velocity() const {
            static double output[3];
            for (int i = 0; i < 3; ++i) {
                output[i] = _velocity[i];
            }
            return output;
        }

        // Set velocity on device
        CUDA_DEV inline void velocity(double * v) {
            for (int i = 0; i < 3; ++i) {
                _velocity[i] = v[i];
            }
        }

        // Set velocity on host
        CUDA_HOST inline void velocity_h(const cartesian_t & v) {
            for (int i = 0; i < 3; ++i) {
                _velocity[i] = v[i];
            }
        }

        // Get epoch UTC time
        CUDA_DEV inline double time() const { return _t; }

        // Set epoch UTC time on device
        CUDA_DEV inline void time(double t) { _t = t; } 

        // Set epoch UTC time on host
        CUDA_HOST inline void time_h(double t) { _t = t; }

    private:
        double _position[3];
        double _velocity[3];
        double _t;
};

#endif

// end of file
