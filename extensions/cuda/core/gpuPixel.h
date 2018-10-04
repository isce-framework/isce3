//
// Author: Bryan V. Riel
// Copyright 2017-2018
//

#ifndef ISCE_CUDA_CORE_GPUPIXEL_H
#define ISCE_CUDA_CORE_GPUPIXEL_H

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
#include "isce/core/Pixel.h"

// Declaration
namespace isce {
    namespace cuda {
        namespace core {
            class gpuPixel;
        }
    }
}

class isce::cuda::core::gpuPixel {

    public:
        // Constructors
        CUDA_HOSTDEV gpuPixel() {};
        CUDA_HOSTDEV gpuPixel(double r, double d, size_t b) : _range(r), _dopfact(d), _bin(b) {}
        CUDA_HOST gpuPixel(const isce::core::Pixel & pixel) :
            _range(pixel.range()), _dopfact(pixel.dopfact()), _bin(pixel.bin()) {}

        // Getters
        CUDA_DEV double range() const { return _range; }
        CUDA_DEV double dopfact() const { return _dopfact; }
        CUDA_DEV size_t bin() const { return _bin; }

        // Setters
        CUDA_DEV void range(double r) { _range = r; }
        CUDA_DEV void dopfact(double d) { _dopfact = d; }
        CUDA_DEV void bin(size_t b) { _bin = b; }

    private:
        double _range;
        double _dopfact;
        size_t _bin;
};

#endif

// end of file
