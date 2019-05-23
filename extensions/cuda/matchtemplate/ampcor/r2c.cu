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
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"

// helpers
__global__
static void
_r2c(const float * gamma, std::size_t corDim, std::size_t zmdDim, cuComplex * scratch);


// compute the amplitude of the signal tiles, assuming pixels are of type std::complex<float>
auto
ampcor::cuda::kernels::
r2c(const float * gamma,
    std::size_t pairs,
    std::size_t corDim, std::size_t zmdDim) -> cuComplex *
{
    // constants
    const auto Mb = 1.0 / 1024 / 1024;
    // grab a spot
    cuComplex * scratch = nullptr;
    // compute the number of cells in the zoomed correlation matrix
    auto zmdCells = zmdDim * zmdDim;
    // compute the amount of memory we need
    auto footprint = pairs * zmdCells * sizeof(cuComplex);
    // allocate memory for the complex zoomed version
    auto status = cudaMallocManaged(&scratch, footprint);
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
    // initialize the memory
    status = cudaMemset(scratch, 0, footprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while initializing " << footprint * Mb
            << " Mb of device memory for the zoomed correlation matrix: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }

    // we will launch enough blocks
    auto B = pairs;
    // with enough threads
    auto T = 32 * (corDim / 32 + (corDim % 32) ? 1 : 0);

    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each to process "
        << corDim << " columns of the correlation hyper-matrix"
        << pyre::journal::endl;

    // launch
    _r2c <<<B,T>>> (gamma, corDim, zmdDim, scratch);
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
            << "while upcasting and embedding the correlation hyper-matrix: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

    // all done
    return scratch;
}


// implementations
__global__
static void
_r2c(const float * gamma, std::size_t corDim, std::size_t zmdDim, cuComplex * scratch)
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
    if (t >= corDim) {
        // nothing to do
        return;
    }

    // compute the number of cells in the correlation matrix
    auto corCells = corDim * corDim;
    // compute the number of cells in a zoomed correlation matrix
    auto zmdCells = zmdDim * zmdDim;

    // find the matrix I'm reading from and skip to my column
    auto src = gamma + b * corCells + t;
    // and the matrix I'm writing to and skip to my column
    auto dst = scratch + b * zmdCells + t;

    // transfer one whole column of {gamma} to {scratch}
    for (auto idx = 0; idx < corDim; ++idx) {
        // read the data, convert to complex, and store
        *dst = {*src, 0};
        // update the pointers:
        // {src} skips {corDim} cells
        src += corDim;
        // while {dst} must skip over {zmdDim} cells;
        dst += zmdDim;
    }

    // all done
    return;
}


// end of file
