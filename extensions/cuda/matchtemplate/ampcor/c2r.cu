// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//


// configuration
//#include <portinfo>
// STL
#include <exception>
#include <string>
// cuda
#include <cuda_runtime.h>
#include <cuComplex.h>
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"

// helpers
__global__
static void
_c2r(const cuComplex * scratch, std::size_t zmdDim, float * zoomed);


// compute the amplitude of the signal tiles, assuming pixels are of type std::complex<float>
auto
ampcor::cuda::kernels::
c2r(const cuComplex * scratch, std::size_t pairs, std::size_t zmdDim) -> float *
{
    // constants
    const auto Mb = 1.0 / 1024 / 1024;
    // grab a spot
    float * zoomed = nullptr;
    // compute the number of cells
    auto zmdCells = zmdDim * zmdDim;
    // and the amount of memory required
    auto footprint = pairs * zmdCells * sizeof(float);
    // allocate memory for the hyper-matrix
    auto status = cudaMallocManaged(&zoomed, footprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while allocating " << footprint * Mb
            << " Mb of device memory for the zoomed correlation matrix"
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::bad_alloc();
    }

    // we will launch enough blocks
    auto B = pairs;
    // with enough threads
    auto T = 32 * (zmdDim / 32 + (zmdDim % 32) ? 1 : 0);

    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each to process "
        << zmdDim << " columns of the correlation hyper-matrix"
        << pyre::journal::endl;

    // launch
    _c2r <<<B,T>>> (scratch, zmdDim, zoomed);
    // wait for the device to finish
    status = cudaDeviceSynchronize();
    // if something went wrong
    if (status != cudaSuccess) {
        // form the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while detecting the zoomed correlation hyper-matrix: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

    // all done
    return zoomed;
}


// implementations
__global__
static void
_c2r(const cuComplex * scratch, std::size_t zmdDim, float * zoomed)
{
    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;      // number of blocks
    // std::size_t T = blockDim.x;        // number of threads per block
    // auto W = B*T;                   // total number of workers
    // local
    std::size_t b = blockIdx.x;        // my block id
    std::size_t t = threadIdx.x;       // my thread id
    // auto w = b*T + t;               // my worker id

    // if there is no work for me
    if (t >= zmdDim) {
        // nothing to do
        return;
    }

    // compute the number of cells in a zoomed correlation matrix
    auto zmdCells = zmdDim * zmdDim;

    // find the matrix I'm reading from and skip to my column
    auto src = scratch + b * zmdCells + t;
    // and the matrix I'm writing to and skip to my column
    auto dst = zoomed + b * zmdCells + t;

    // transfer one whole column of {gamma} to {scratch}
    for (auto idx = 0; idx < zmdDim; ++idx) {
        // read the data, convert to complex, and store
        *dst = cuCabsf(*src);
        // update the pointers:
        // {src} skips {zmdDim} cells
        src += zmdDim;
        // while {dst} must skip over {zmdDim} cells;
        dst += zmdDim;
    }

    // all done
    return;
}


// end of file
