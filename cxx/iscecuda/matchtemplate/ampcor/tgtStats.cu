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
// pull the declarations
#include "kernels.h"


// the SAT generation kernel
template <typename value_t = float>
__global__
static void
_tgtStats(const value_t * sat,
          std::size_t tiles, std::size_t refDim, std::size_t tgtDim, std::size_t corDim,
          value_t * stats);


// implementation

// precompute the amplitude averages for all possible placements of the search tile within the
// target search window for all pairs in the plan. we allocate room for {_pairs}*{_corCells}
// floating point values and use the precomputed SAT tables resident on the device.
//
// the SAT tables require a slice and produce the sum of the values of cells within the slice
// in no more than four memory accesses per search tile; there are boundary cases to consider
// that add a bit of complexity to the implementation; the boundary cases could have been
// trivialized using ghost cells around the search window boundary, but the memory cost is high
void
ampcor::cuda::kernels::
tgtStats(const float * dSAT,
         std::size_t pairs, std::size_t refDim, std::size_t tgtDim, std::size_t corDim,
         float * dStats)
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
        << " threads each to handle the " << pairs
        << " entries of the hyper-grid of target amplitude averages"
        << pyre::journal::endl;
    // launch the kernels
    _tgtStats <<<B,T>>> (dSAT, pairs, refDim, tgtDim, corDim, dStats);
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
            << "while computing the average amplitudes of all possible search window placements: "
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
_tgtStats(const value_t * dSAT,
      std::size_t tiles,     // the total number of target tiles
      std::size_t refDim,    // the shape of each reference tile
      std::size_t tgtDim,    // the shape of each target tile
      std::size_t corDim,    // the shape of each grid
      value_t * dStats)
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
    if (w >= tiles) {
        // nothing for me to do
        return;
    }

    // compute the number of cells in a reference tile
    auto refCells = refDim * refDim;
    // compute the number of cells in a target tile
    auto tgtCells = tgtDim * tgtDim;
    // compute the number of cells in each correlation matrix
    auto corCells = corDim * corDim;

    // locate the beginning of my SAT table
    auto sat = dSAT + w*tgtCells;
    // locate the beginning of my stats table
    auto stats = dStats + w*corCells;

    // go through all possible row offsets
    for (auto row = 0; row < corDim; ++row) {
        // the row limit of the tile
        // this depends on the shape of the reference tile
        auto rowMax = row + refDim - 1;
        // go through all possible column offsets
        for (auto col = 0; col < corDim; ++col) {
            // the column limit of the tile
            //  this depends on the shape of the reference tile
            auto colMax = col + refDim - 1;

            // initialize the sum by reading the bottom right corner; it's guaranteed to be
            // within the SAT
            auto sum = sat[rowMax*tgtDim + colMax];

            // if the slice is not top-aligned
            if (row > 0) {
                // subtract the value from the upper right corner
                sum -= sat[(row-1)*tgtDim + colMax];
            }

            // if the slice is not left-aligned
            if (col > 0) {
                // subtract the value of the upper left corner
                sum -= sat[rowMax*tgtDim + (col - 1)];
            }

            // if the slice is not aligned with the upper left corner
            if (row > 0 && col > 0) {
                // restore its contribution to the sum
                sum += sat[(row-1)*tgtDim + (col-1)];
            }

            // compute the offset that brings us to this placement in this tile
            auto offset = row*corDim + col;
            // compute the average value and store it
            stats[offset] = sum / refCells;
        }
    }

    // all done
    return;
}


// end of file
