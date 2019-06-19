//
// Author: Bryan Riel
// Copyright 2018
//

#ifndef ISCE_CUDA_GEOMETRY_GPUTOPOLAYERS_H
#define ISCE_CUDA_GEOMETRY_GPUTOPOLAYERS_H

#include <iostream>

#include <isce/core/Common.h>
#include <isce/geometry/TopoLayers.h>

// Declaration
namespace isce {
    namespace cuda {
        namespace geometry {
            class gpuTopoLayers;
        }
    }
}

// DEMInterpolator declaration
class isce::cuda::geometry::gpuTopoLayers {

    public:
        // Constructor for host only - allocate memory on device
        CUDA_HOST gpuTopoLayers(const isce::geometry::TopoLayers & layers);
    
        // Copy constructor on device (these should nominally be CUDA_HOSTDEV)
        CUDA_HOSTDEV inline gpuTopoLayers(gpuTopoLayers & layers) :
            _length(layers.length()), _width(layers.width()), _x(layers._x),
            _y(layers._y), _z(layers._z), _inc(layers._inc), _hdg(layers._hdg),
            _localInc(layers._localInc), _localPsi(layers._localPsi), _sim(layers._sim),
            _crossTrack(layers._crossTrack), _nbytes_double(layers.nbytes_double()),
            _nbytes_float(layers.nbytes_float()), _owner(false) {}

        // Destructor
        CUDA_HOST ~gpuTopoLayers();

        // Set values for a single index; on GPU, all arrays are flattened
        CUDA_DEV inline void x(size_t index, double value) {
            _x[index] = value;
        }

        CUDA_DEV inline void y(size_t index, double value) {
            _y[index] = value;
        }

        CUDA_DEV inline void z(size_t index, double value) {
            _z[index] = value;
        }

        CUDA_DEV inline void inc(size_t index, float value) {
            _inc[index] = value;
        }

        CUDA_DEV inline void hdg(size_t index, float value) {
            _hdg[index] = value;
        }

        CUDA_DEV inline void localInc(size_t index, float value) {
            _localInc[index] = value;
        }

        CUDA_DEV inline void localPsi(size_t index, float value) {
            _localPsi[index] = value;
        }

        CUDA_DEV inline void sim(size_t index, float value) {
            _sim[index] = value;
        }

        CUDA_DEV inline void crossTrack(size_t index, double value) {
            _crossTrack[index] = value;
        }

        // Get sizes on host or device
        CUDA_HOSTDEV inline size_t length() const { return _length; }
        CUDA_HOSTDEV inline size_t width() const { return _width; }
        CUDA_HOSTDEV inline size_t nbytes_double() const { return _nbytes_double; }
        CUDA_HOSTDEV inline size_t nbytes_float() const { return _nbytes_float; }

        // Copy results to host TopoLayers
        CUDA_HOST void copyToHost(isce::geometry::TopoLayers & layers);

        // Unlike CPU version, make the data pointers public to allow for easy
        // copy construction on the device; still use the underbar convention to discourage
        // interfacing with pointers directly
        double * _x;
        double * _y;
        double * _z;
        float * _inc;
        float * _hdg;
        float * _localInc;
        float * _localPsi;
        float * _sim;
        double * _crossTrack;

    private:
        size_t _length;
        size_t _width;
        size_t _nbytes_double, _nbytes_float;
        bool _owner;
};

#endif

// end of file
