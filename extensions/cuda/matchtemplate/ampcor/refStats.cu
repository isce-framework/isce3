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
#include <complex>
#include <string>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"

// helpers
template <std::size_t T, typename value_t = float>
__global__
void
_refStats(value_t * rArena,
          std::size_t refDim, std::size_t cellsPerTilePair,
          value_t * stats);


// compute the amplitude of the signal tiles, assuming pixels are of type std::complex<float>
void
ampcor::cuda::kernels::
refStats(float * rArena,
         std::size_t pairs, std::size_t refDim, std::size_t cellsPerTilePair,
         float * stats)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "arena has " << pairs << " blocks of " << cellsPerTilePair << " cells;"
        << " the reference tiles are " << refDim << "x" << refDim
        << pyre::journal::endl;

    // the number of threads per block
    auto T = refDim;
    // the number of blocks
    auto B = pairs;
    // the amount of shared memory
    auto S = std::max(64ul, T) * sizeof(float);

    // show me
    channel << pyre::journal::at(__HERE__);
    // deploy
    if (refDim <= 32) {
        // show me
        channel << "deploying the 32x32 kernel";
        // with 32x32 tiles
        _refStats<32><<<B, 32, S>>>(rArena, refDim, cellsPerTilePair, stats);
    } else if (refDim <= 64) {
        // show me
        channel << "deploying the 64x64 kernel";
        // with 64x64 tiles
        _refStats<64><<<B, 64, S>>>(rArena, refDim, cellsPerTilePair, stats);
    } else if (refDim <= 128) {
        // show me
        channel << "deploying the 128x128 kernel";
        // with 128x128 tiles
        _refStats<128><<<B, 128, S>>>(rArena, refDim, cellsPerTilePair, stats);
    } else if (refDim <= 256) {
        // show me
        channel << "deploying the 256x256 kernel";
        // with 256x256 tiles
        _refStats<256><<<B, 256, S>>>(rArena, refDim, cellsPerTilePair, stats);
    } else if (refDim <= 512) {
        // show me
        channel << "deploying the 512x512 kernel";
        // with 512x512 tiles
        _refStats<512><<<B, 512, S>>>(rArena, refDim, cellsPerTilePair, stats);
    } else {
        // complain
        throw std::runtime_error("cannot handle reference tiles of this shape");
    }
    // flush
    channel << pyre::journal::endl;

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
            << "while ensuring the detected reference tiles have zero mean: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// implementations
template <std::size_t T, typename value_t>
__global__
void
_refStats(value_t * rArena,
          std::size_t refDim, std::size_t cellsPerTilePair,
          value_t * stats)
{
    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;      // number of blocks
    // std::size_t T = blockDim.x;     // number of threads per block
    // auto W = B*T;                   // total number of workers
    // local
    std::size_t b = blockIdx.x;        // my block id
    std::size_t t = threadIdx.x;       // my thread id
    // auto w = b*T + t;               // my worker id

    // N.B.: do not be tempted to terminate early threads that have no assigned workload; their
    // participation is required to make sure that shared memory is properly zeored out for the
    // nominally out of bounds accesses

    // access to my shared memory
    extern __shared__ value_t scratch[];
    // handle to my thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // step one: every thread sums a column of its tile
    // find the start of my tile by skipping the tile pairs handled by the lesser blocks
    auto tile = rArena + b*cellsPerTilePair;
    // compute the location of the cell past the end of my tile
    auto eot = tile + refDim*refDim;
    // initialize the accumulator
    value_t sum = 0;
    // if my thread id is less than the number of columns, i need to sum up the values;
    // otherwise, my contribution is to zero out my slot in shared memory
    if (t < refDim) {
        // run down my column
        for (auto cell = tile + t; cell < eot; cell += refDim) {
            // picking up contributions
            sum += *cell;
        }
    }
    // store the partial sum in my slot in shared memory
    scratch[t] = sum;
    // make sure everybody is done
    cta.sync();

    // step two: reduction in shared memory
    // for progressively smaller block sizes, the bottom half of the threads collect partial sums
    // N.B.: T is a template parameter, known at compile time, so it's easy for the optimizer to
    // eliminate the impossible clauses
    // for 512 threads per block
    if (T >= 512 && t < 256) {
        // update my partial sum by reading my sibling's value
        sum += scratch[t + 256];
        // and make it available in my shared memory slot
        scratch[t] = sum;
    }
    // make sure everybody is done
    cta.sync();
    // for 256 threads per block
    if (T >= 256 && t < 128) {
        // update my partial sum by reading my sibling's value
        sum += scratch[t + 128];
        // and make it available in my shared memory slot
        scratch[t] = sum;
    }
    // make sure everybody is done
    cta.sync();
    // for 128 threads per block
    if (T >= 128 && t < 64) {
        // update my partial sum by reading my sibling's value
        sum += scratch[t + 64];
        // and make it available in my shared memory slot
        scratch[t] = sum;
    }
    // make sure everybody is done
    cta.sync();
    // on recent architectures, there is a faster way to do the reduction once we reach the
    // warp level; the only cost is that we have to make sure there is enough memory for 64
    // threads, i.e. the shared memory size is bound from below by 64*sizeof(value_t)
    if (t < 32) {
        // if we need to
        if (T >= 64) {
            // update the partial sum from the second warp
            sum += scratch[t + 32];
        }

        // grab the block of active threads
        cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();

        // the threads with power of 2 ids
        for (auto offset = 16; offset > 0; offset >>= 1) {
            // reduce using a warp shuffle
            sum += active.shfl_down(sum, offset);
        }
    }
    // finally, thread 0
    if (t == 0) {
        // saves the final value
        scratch[0] = sum / (refDim*refDim);
    }
    // make sure everybody is done
    cta.sync();

    // step three: revisit the tile and subtract this value from all cells, and accumulate the
    // sum of the squares of the resulting cells so we can compute the variance; again, do not
    // be tempted to send idle threads home
    // initialize the variance
    value_t sumsq = 0;
    // only threads assigned to columns do any work
    if (t < refDim) {
        // read the mean value from shared memory
        auto mean = scratch[0];
        // run down my column
        for (auto cell = tile + t; cell < eot; cell += refDim) {
            // get the cell value and subtract the mean
            auto value = *cell - mean;
            // store it
            *cell = value;
            // update the sum of the squares
            sumsq += value*value;
        }
    }
    // store the partial sum in my slot in shared memory
    scratch[t] = sumsq;
    // make sure everybody is done
    cta.sync();

    // step four: reduce the sum of the squares to compute the variance
    // for 512 threads per block
    if (T >= 512 && t < 256) {
        // update my partial sum by reading my sibling's value
        sumsq += scratch[t + 256];
        // and make it available in my shared memory slot
        scratch[t] = sumsq;
    }
    // make sure everybody is done
    cta.sync();
    // for 256 threads per block
    if (T >= 256 && t < 128) {
        // update my partial sum by reading my sibling's value
        sumsq += scratch[t + 128];
        // and make it available in my shared memory slot
        scratch[t] = sumsq;
    }
    // make sure everybody is done
    cta.sync();
    // for 128 threads per block
    if (T >= 128 && t < 64) {
        // update my partial sum by reading my sibling's value
        sumsq += scratch[t + 64];
        // and make it available in my shared memory slot
        scratch[t] = sumsq;
    }
    // make sure everybody is done
    cta.sync();
    // on recent architectures, there is a faster way to do the reduction once we reach the
    // warp level; the only cost is that we have to make sure there is enough memory for 64
    // threads, i.e. the shared memory size is bound from below by 64*sizeof(value_t)
    if (t < 32) {
        // if we need to
        if (T >= 64) {
            // update the partial sum from the second warp
            sumsq += scratch[t + 32];
        }

        // grab the block of active threads
        cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();

        // the threads with power of 2 ids
        for (auto offset = 16; offset > 0; offset >>= 1) {
            // reduce using a warp shuffle
            sumsq += active.shfl_down(sumsq, offset);
        }
    }
    // finally, thread 0
    if (t == 0) {
        // computes the variance
        auto var = std::sqrt(sumsq);
        // and saves it in the output array; recall: one block per tile, so the correct memory
        // location to save the answer is given by my block id
        stats[b] = var;
    }

    // all done
    return;
}


// end of file
