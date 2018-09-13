//
// Author: Bryan Riel
// Copyright 2018
//

#ifndef ISCE_CUDA_GEOMETRY_GPUTOPOLAYERS_H
#define ISCE_CUDA_GEOMETRY_GPUTOPOLAYERS_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#endif

#include <iostream>
#include <helper_cuda.h>

// isce::geometry
#include "isce/geometry/TopoLayers.h"

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
        CUDA_HOST inline gpuTopoLayers(const isce::geometry::TopoLayers & layers) : 
            _length(layers.length()), _width(layers.width()), _owner(true) {

            // Specify the device
            cudaSetDevice(0);

            // Allocate memory
            _nbytes_double = _length * _width * sizeof(double);
            _nbytes_float = _length * _width * sizeof(float);
            checkCudaErrors(cudaMalloc((double **) &_x, _nbytes_double));
            checkCudaErrors(cudaMalloc((double **) &_y, _nbytes_double));
            checkCudaErrors(cudaMalloc((double **) &_z, _nbytes_double));
            checkCudaErrors(cudaMalloc((float **) &_inc, _nbytes_float));
            checkCudaErrors(cudaMalloc((float **) &_hdg, _nbytes_float));
            checkCudaErrors(cudaMalloc((float **) &_localInc, _nbytes_float));
            checkCudaErrors(cudaMalloc((float **) &_localPsi, _nbytes_float));
            checkCudaErrors(cudaMalloc((float **) &_sim, _nbytes_float));
        }
    
        // Copy constructor on device (these should nominally be CUDA_HOSTDEV)
        CUDA_HOSTDEV inline gpuTopoLayers(gpuTopoLayers & layers) :
            _length(layers.length()), _width(layers.width()), _x(layers._x),
            _y(layers._y), _z(layers._z), _inc(layers._inc), _hdg(layers._hdg),
            _localInc(layers._localInc), _localPsi(layers._localPsi), _sim(layers._sim),
            _nbytes_double(layers.nbytes_double()), _nbytes_float(layers.nbytes_float()),
            _owner(false) {}

        // Destructor
        inline ~gpuTopoLayers() {
            if (_owner) {
                checkCudaErrors(cudaFree(_x));
                checkCudaErrors(cudaFree(_y));
                checkCudaErrors(cudaFree(_z));
                checkCudaErrors(cudaFree(_inc));
                checkCudaErrors(cudaFree(_hdg)); 
                checkCudaErrors(cudaFree(_localInc));
                checkCudaErrors(cudaFree(_localPsi));
                checkCudaErrors(cudaFree(_sim));
            }
        }

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

        // Get sizes on host or device
        CUDA_HOSTDEV inline size_t length() const { return _length; }
        CUDA_HOSTDEV inline size_t width() const { return _width; }
        CUDA_HOSTDEV inline size_t nbytes_double() const { return _nbytes_double; }
        CUDA_HOSTDEV inline size_t nbytes_float() const { return _nbytes_float; }

        // Copy results to host TopoLayers
        CUDA_HOST inline void copyToHost(isce::geometry::TopoLayers & layers) {
            checkCudaErrors(cudaMemcpy(&layers.x()[0], _x, _nbytes_double,
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&layers.y()[0], _y, _nbytes_double,
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&layers.z()[0], _z, _nbytes_double,
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&layers.inc()[0], _inc, _nbytes_float,
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&layers.hdg()[0], _hdg, _nbytes_float,
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&layers.localInc()[0], _localInc, _nbytes_float,
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&layers.localPsi()[0], _localPsi, _nbytes_float,
                            cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&layers.sim()[0], _sim, _nbytes_float,
                            cudaMemcpyDeviceToHost));
        }

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

    private:
        size_t _length;
        size_t _width;
        size_t _nbytes_double, _nbytes_float;
        bool _owner;
};

#endif

// end of file
