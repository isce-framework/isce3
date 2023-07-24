#include "gpuTopoLayers.h"

#include <cuda_runtime.h>
#include <isce3/geometry/TopoLayers.h>
#include <isce3/cuda/except/Error.h>

namespace isce3::cuda::geometry {
    gpuTopoLayers::gpuTopoLayers(const isce3::geometry::TopoLayers & layers) :
        _length(layers.length()), _width(layers.width()), _owner(true) {

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
        checkCudaErrors(cudaMalloc((double **) &_crossTrack, _nbytes_double));
        checkCudaErrors(cudaMalloc((float **) &_groundToSatEast,
                    _nbytes_float));
        checkCudaErrors(cudaMalloc((float **) &_groundToSatNorth,
                    _nbytes_float));
    }

    // Destructor
    gpuTopoLayers::~gpuTopoLayers() {
        if (_owner) {
            checkCudaErrors(cudaFree(_x));
            checkCudaErrors(cudaFree(_y));
            checkCudaErrors(cudaFree(_z));
            checkCudaErrors(cudaFree(_inc));
            checkCudaErrors(cudaFree(_hdg));
            checkCudaErrors(cudaFree(_localInc));
            checkCudaErrors(cudaFree(_localPsi));
            checkCudaErrors(cudaFree(_sim));
            checkCudaErrors(cudaFree(_crossTrack));
            checkCudaErrors(cudaFree(_groundToSatEast));
            checkCudaErrors(cudaFree(_groundToSatNorth));
        }
    }

    // Copy results to host TopoLayers
    void gpuTopoLayers::copyToHost(isce3::geometry::TopoLayers & layers) {

        /*
        Check whether memory for each layer has been allocated by
        CPU TopoLayers before copying corresponding data from GPU to the CPU.

        Note that it's not necessary to check whether the GPU memory has
        also been allocated since gpuTopoLayers allocates memory for all
        arrays (unlike CPU TopoLayers)
        */

        if (layers.x().size()) {
            checkCudaErrors(cudaMemcpy(&layers.x()[0], _x, _nbytes_double,
                            cudaMemcpyDeviceToHost));
        }
        if (layers.y().size()) {
            checkCudaErrors(cudaMemcpy(&layers.y()[0], _y, _nbytes_double,
                            cudaMemcpyDeviceToHost));
        }
        if (layers.z().size()) {
            checkCudaErrors(cudaMemcpy(&layers.z()[0], _z, _nbytes_double,
                            cudaMemcpyDeviceToHost));
        }
        if (layers.inc().size()) {
            checkCudaErrors(cudaMemcpy(&layers.inc()[0], _inc, _nbytes_float,
                            cudaMemcpyDeviceToHost));
        }
        if (layers.hdg().size()) {
            checkCudaErrors(cudaMemcpy(&layers.hdg()[0], _hdg, _nbytes_float,
                            cudaMemcpyDeviceToHost));
        }
        if (layers.localInc().size()) {
            checkCudaErrors(cudaMemcpy(&layers.localInc()[0], _localInc, _nbytes_float,
                            cudaMemcpyDeviceToHost));
        }
        if (layers.localPsi().size()) {
            checkCudaErrors(cudaMemcpy(&layers.localPsi()[0], _localPsi, _nbytes_float,
                            cudaMemcpyDeviceToHost));
        }
        if (layers.sim().size()) {
            checkCudaErrors(cudaMemcpy(&layers.sim()[0], _sim, _nbytes_float,
                            cudaMemcpyDeviceToHost));
        }
        if (layers.crossTrack().size()) {
            checkCudaErrors(cudaMemcpy(&layers.crossTrack()[0], _crossTrack, _nbytes_double,
                            cudaMemcpyDeviceToHost));
        }
        if (layers.hasGroundToSatEastRaster()) {
            checkCudaErrors(cudaMemcpy(&layers.groundToSatEast()[0],
                        _groundToSatEast, _nbytes_float,
                        cudaMemcpyDeviceToHost));
        }
        if (layers.hasGroundToSatNorthRaster()) {
            checkCudaErrors(cudaMemcpy(&layers.groundToSatNorth()[0],
                        _groundToSatNorth, _nbytes_float,
                        cudaMemcpyDeviceToHost));
        }
    }
}
