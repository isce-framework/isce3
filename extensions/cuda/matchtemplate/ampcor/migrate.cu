// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//


// configuration
//#include <portinfo>
// pyre
#include <pyre/journal.h>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// pull the declarations
#include "kernels.h"


// the correlation kernel
template <typename pixel_t = cuComplex>
__global__
void
_migrate(const pixel_t * coarse,
         std::size_t cellsPerPair, std::size_t cellsPerRefinedPair,
         std::size_t refCells, std::size_t tgtCells,
         std::size_t refRefinedCells, std::size_t tgtRefinedCells,
         std::size_t rdim, std::size_t tdim, std::size_t edim, std::size_t trdim,
         const int * locations,
         pixel_t * refined);


// implementation
void
ampcor::cuda::kernels::
migrate(const std::complex<float> * coarse,
        std::size_t pairs,
        std::size_t refDim, std::size_t tgtDim, std::size_t expDim,
        std::size_t refRefinedDim, std::size_t tgtRefinedDim,
        const int * locations,
        std::complex<float> * refined)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");

    // figure out the job layout and launch the calculation on the device
    // each thread block takes care of one tile pair, so we need as many blocks as there are pairs
    auto B = pairs;
    // the number of threads per block is determined by the shape of the expanded maxcor tile;
    // we round up to the next warp
    auto T = 32 * (expDim / 32 + (expDim % 32 ? 1 : 0));
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each"
        << "to migrate the expanded maxcor tiles to the refinement arena"
        << pyre::journal::endl;

    // shape calculations
    auto refCells = refDim * refDim;
    auto tgtCells = tgtDim * tgtDim;
    auto refRefinedCells = refRefinedDim * refRefinedDim;
    auto tgtRefinedCells = tgtRefinedDim * tgtRefinedDim;

    // so i can skip over work others are doing
    auto cellsPerPair = refCells + tgtCells;
    auto cellsPerRefinedPair = refRefinedCells + tgtRefinedCells;

    // launch
    _migrate <<<B,T>>> (reinterpret_cast<const cuComplex *>(coarse),
                        cellsPerPair, cellsPerRefinedPair,
                        refCells, tgtCells, refRefinedCells, tgtRefinedCells,
                        refDim, tgtDim, expDim, tgtRefinedDim,
                        locations,
                        reinterpret_cast<cuComplex *>(refined));
    // wait for the device to finish
    auto status = cudaDeviceSynchronize();
    // check
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while migrating the maxcor tiles to the refinement arena: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// the correlation kernel
template <typename pixel_t>
__global__
void
_migrate(const pixel_t * coarse,
         std::size_t cellsPerPair, std::size_t cellsPerRefinedPair,
         std::size_t refCells, std::size_t tgtCells,
         std::size_t refRefinedCells, std::size_t tgtRefinedCells,
         std::size_t rdim, std::size_t tdim, std::size_t edim, std::size_t trdim,
         const int * locations,
         pixel_t * refined)
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

    // each thread transfers column {t} of the expanded maxcor tile from pair {b} to the
    // refinement area

    // if there is no work for me
    if (t >= edim) {
        // bail
        return;
    }

    // unpack the location of the ULHC of my maxcor tile
    auto row = locations[2*b];
    auto col = locations[2*b + 1];

    // the source: (row, col) of the target tile of pair {b} in the coarse arena
    const pixel_t * src = coarse + b*cellsPerPair + refCells + row*tdim + col + t;
    // the destination: the target tile of pair {b} in the refined arena
    pixel_t * dest = refined + b*cellsPerRefinedPair + refRefinedCells + t;

    // printf("thread [b=%lu,t=%lu]: loc=(%d,%d)\n", b, t, row, col);

    // go down the columns in tandem
    for (auto jdx = 0; jdx < edim; ++jdx) {
        // move the data
        *dest = *src;
        // update the pointers
        // source moves by a whole row in the target tile
        src += tdim;
        // destination moves by a whole row in the refined target tile
        dest += trdim;
    }
    // all done
    return;
}


// end of file
