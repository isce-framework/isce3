//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#ifndef ISCE_CUDA_CORE_GPUSTATEVECTOR_H
#define ISCE_CUDA_CORE_GPUSTATEVECTOR_H

// isce::core
#include <isce/core/StateVector.h>

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

    typedef isce::core::Vec3 Vec3;

    public:
        // Constructors
        CUDA_HOSTDEV inline gpuStateVector() {}
        CUDA_HOST inline gpuStateVector(const isce::core::StateVector& state) {
            position_h(state.position());
            velocity_h(state.velocity());
            time_h(state.date().secondsOfDay());
        }

        // Make data public to simplify access and to reduce use of temp variables
        // in device code
        Vec3 _position;
        Vec3 _velocity;
        double time;

        CUDA_HOSTDEV Vec3 position() const { return _position; }
        CUDA_HOSTDEV Vec3 velocity() const { return _velocity; }

        // Set position on host
        CUDA_HOST inline void position_h(const cartesian_t & p) {
            for (int i = 0; i < 3; ++i) {
                _position[i] = p[i];
            }
        }

        // Set velocity on host
        CUDA_HOST inline void velocity_h(const cartesian_t & v) {
            for (int i = 0; i < 3; ++i) {
                _velocity[i] = v[i];
            }
        }

        // Set epoch UTC time on host
        CUDA_HOST inline void time_h(double t) { time = t; }
};

#endif

// end of file
