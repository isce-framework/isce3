// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//


// configuration
//#include <portinfo>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// pyre
#include <pyre/journal.h>
// pull the declarations
#include "kernels.h"


// the SAT generation kernel
template <typename value_t = float>
__global__
static void
_nudge(std::size_t pairs,     // the total number of tiles
       std::size_t oldDim,    // the old shape of the target tiles
       std::size_t newDim,    // the new shape of the target tiles
       std::size_t margin,    // the new margin of the search window
       int * loc);


// run through the correlation matrix for each, find its maximum value and record its location
void
ampcor::cuda::kernels::
nudge(std::size_t pairs,     // the total number of tiles
      std::size_t refDim,    // the shape of the reference tiles
      std::size_t tgtDim,    // the shape of the target tiles
      std::size_t margin,    // the new margin around the reference tile
      int * loc)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");

    // launch blocks of T threads
    auto T = 128;
    // in as many blocks as it takes to handle all pairs
    auto B = pairs / T + (pairs % T ? 1 : 0);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T
        << " threads each to nudge the " << pairs
        << " maxima locations"
        << pyre::journal::endl;
    // launch the kernels
    _nudge <<<B,T>>> (pairs, tgtDim, refDim+2*margin, margin, loc);
    // wait for the kernels to finish
    cudaError_t status = cudaDeviceSynchronize();
    // check
    if (status != cudaSuccess) {
        // get the description of the error
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while nudging the new target tile locations: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// the SAT generation kernel
template <typename value_t>
__global__
void
_nudge(std::size_t pairs,     // the total number of tiles
       std::size_t oldDim,    // the shape of the target tiles
       std::size_t newDim,    // the shape of the target tiles
       std::size_t margin,    // the new margin of the search window
       int * loc)
{
    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;    // number of blocks
    std::size_t T = blockDim.x;      // number of threads per block
    // std::size_t W = B*T;          // total number of workers
    // local
    std::size_t b = blockIdx.x;      // my block id
    std::size_t t = threadIdx.x;     // my thread id within my block
    std::size_t w = b*T + t;         // my worker id

    // if my worker id exceeds the number of cells that require update
    if (w >= pairs) {
        // nothing for me to do
        return;
    }

    // locate the beginning of my stats table
    auto myloc = loc + 2*w;
    // read my position
    int row = myloc[0];
    int col = myloc[1];

    // let's do LR nudging first
    int left = col - margin;
    // if it sticks out in the left
    if (left < 0) {
        // move it to the far left
        left = 0;
    }
    // if it sticks out on the right
    if (left + newDim > oldDim) {
        // move so that it fits
        left = oldDim - newDim;
    }
    // repeat for TB
    int top = row - margin;
    // if it sticks outside the top of the tile
    if (top < 0) {
        // move it to the top row
        top = 0;
    }
    // if it sticks out below the bottom row
    if (top + newDim > oldDim) {
        // move it up so it fits
        top = oldDim - newDim;
    }

    // write the new locations
    myloc[0] = top;
    myloc[1] = left;

    // all done
    return;
}


// end of file
