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
#include <complex>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"


// the SAT generation kernel
template <typename value_t = float>
__global__
static void
_sat(const value_t * dArena,
    std::size_t stride, std::size_t rcells, std::size_t tcells, std::size_t tdim,
    value_t * dSAT);


// implementation
void
ampcor::cuda::kernels::
sat(const float * dArena,
    std::size_t pairs, std::size_t refCells, std::size_t tgtCells, std::size_t tgtDim,
    float * dSAT)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");

    // to compute the SAT for each target tile, we launch as many thread blocks as there are
    // target tiles
    std::size_t B = pairs;
    // the number of threads per block is determined by the shape of the target tile; round up
    // to the nearest warp
    std::size_t T = 32 * (tgtDim / 32 + (tgtDim % 32 ? 1 : 0));
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T
        << " threads each to compute SATs for the target tiles"
        << pyre::journal::endl;

    // launch the SAT kernel
    _sat <<<B,T>>> (dArena, refCells+tgtCells, refCells, tgtCells, tgtDim, dSAT);
    // wait for the device to finish
    cudaError_t status = cudaDeviceSynchronize();
    // if something went wrong
    if (status != cudaSuccess) {
        // form the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while computing the sum area tables: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// the SAT generation kernel
template <typename value_t>
__global__
void
_sat(const value_t * dArena,
    std::size_t stride, std::size_t rcells, std::size_t tcells, std::size_t tdim,
    value_t * dSAT)
{
    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;    // number of blocks
    // std::size_t T = blockDim.x;   // number of threads per block
    // std::size_t W = B*T;          // total number of workers
    // local
    std::size_t b = blockIdx.x;      // my block id
    std::size_t t = threadIdx.x;     // my thread id within my block
    // std::size_t w = b*T + t;      // my worker id

    // if there is nothing for me to do
    if (t >= tdim) {
        // bail
        return;
    }

    // get a handle to this thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // on the first pass, each thread sweeps across its row in the target tile
    // on a second pass, each thread sweeps down its column in the SAT

    // across the row
    // my starting point for reading data is row {t} of tile {b} in the arena
    std::size_t read = b*stride + rcells + t*tdim;
    // my starting point for writing data is row {t} of tile {b} in the SAT area
    std::size_t write = b*tcells + t*tdim;

    // initialize the partial sum
    value_t sum = 0;

    // run across the row
    for (auto slot = 0; slot < tdim; ++slot) {
        // update the sum
        sum += dArena[read + slot];
        // store the result
        dSAT[write + slot] = sum;
    }

    // barrier: make sure everybody is done updating the SAT
    cta.sync();

    // march down the column of the SAT table itself
    // my starting point is column {t} of tile {b}
    std::size_t colStart = b*tcells + t;
    // can't go past the end of my tile
    std::size_t colStop = (b+1)*tcells;
    // reinitialize the partial sum
    sum = 0;
    // run
    for (auto slot=colStart; slot < colStop; slot += tdim) {
        // read the current value and save it
        auto current = dSAT[slot];
        // update the current value with the running sum
        dSAT[slot] += sum;
        // update the running sum for the next guy
        sum += current;
    }

    // all done
    return;
}


// end of file
