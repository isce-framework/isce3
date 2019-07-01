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
// pyre
#include <pyre/journal.h>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// pull the declarations
#include "kernels.h"


// the correlation kernel
template <std::size_t T, typename value_t = float>
__global__
void
_correlate(const value_t * arena,
           const value_t * refStats, const value_t * tgtStats,
           std::size_t rdim, std::size_t rcells,
           std::size_t tdim, std::size_t tcells,
           std::size_t cdim, std::size_t ccells,
           std::size_t row, std::size_t col,
           value_t * correlation);


// implementation
void
ampcor::cuda::kernels::
correlate(const float * dArena, const float * refStats, const float * tgtStats,
          std::size_t pairs,
          std::size_t refCells, std::size_t tgtCells, std::size_t corCells,
          std::size_t refDim, std::size_t tgtDim, std::size_t corDim,
          float * dCorrelation)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");

    // figure out the job layout and launch the calculation on the device
    // each thread block takes care of one tile pair, so we need as many blocks as there are pairs
    auto B = pairs;
    // the number of threads per block is determined by the shape of the reference  tile
    auto T = refDim;
    // each thread stores in shared memory the partial sum for the numerator term and the
    // partial sum for the target tile variance; so we need two {value_t}'s worth of shared
    // memory for each thread
    auto S = 2 * T * sizeof(float);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each, with "
        << S << " bytes of shared memory per block, for each of the " << corCells
        << " possible placements of the search window within the target tile;"
        << " a grand total of " << (B*corCells) << " kernel launches"
        << pyre::journal::endl;

    // for storing error codes
    cudaError_t status = cudaSuccess;
    // go through all possible row offsets for the sliding window
    for (auto row = 0; row < corDim; ++row) {
        // and all possible column offsets
        for (auto col = 0; col < corDim; ++col) {
            // deduce the correct kernel to launch and deploy
            // N.B.: kernel launch is an implicit barrier, so no need for any extra
            // synchronization
            if (refDim <= 32) {
                // tell me
                channel << "deploying the 32x32 kernel";
                // do it
                _correlate<32> <<<B,32,S>>> (dArena, refStats, tgtStats,
                                             refDim, refCells, tgtDim, tgtCells, corDim, corCells,
                                             row, col, dCorrelation);
            } else if (refDim <= 64) {
                // tell me
                channel << "deploying the 64x64 kernel";
                // do it
                _correlate<64> <<<B,64,S>>> (dArena, refStats, tgtStats,
                                             refDim, refCells, tgtDim, tgtCells, corDim, corCells,
                                             row, col, dCorrelation);
            } else if (refDim <= 128) {
                // tell me
                channel << "deploying the 128x128 kernel";
                // do it
                _correlate<128> <<<B,128,S>>> (dArena, refStats, tgtStats,
                                               refDim, refCells, tgtDim, tgtCells, corDim, corCells,
                                               row, col, dCorrelation);
            } else if (refDim <= 256) {
                // tell me
                channel << "deploying the 256x256 kernel";
                // do it
                _correlate<256> <<<B,256,S>>> (dArena, refStats, tgtStats,
                                               refDim, refCells, tgtDim, tgtCells, corDim, corCells,
                                               row, col, dCorrelation);
            } else if (refDim <= 512) {
                // tell me
                channel << "deploying the 512x512 kernel";
                // do it
                _correlate<512> <<<B,512,S>>> (dArena, refStats, tgtStats,
                                               refDim, refCells, tgtDim, tgtCells, corDim, corCells,
                                               row, col, dCorrelation);
            } else {
                // complain
                throw std::runtime_error("cannot handle reference tiles of this shape");
            }
            // check for errors
            status = cudaPeekAtLastError();
            // if something went wrong
            if (status != cudaSuccess) {
                // make a channel
                pyre::journal::error_t error("ampcor.cuda");
                // complain
                error
                    << pyre::journal::at(__HERE__)
                    << "after launching the " << row << "x" << col << " correlators: "
                    << cudaGetErrorName(status) << " (" << status << ")"
                    << pyre::journal::endl;
                // and bail
                break;
            }
        }
        // if something went wrong in the inner loop
        if (status != cudaSuccess) {
            // bail out of the outer loop as well
            break;
        }
    }
    // wait for the device to finish
    status = cudaDeviceSynchronize();
    // check
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while waiting for a kernel to finish: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// the correlation kernel
template <std::size_t T, typename value_t>
__global__
void
_correlate(const value_t * arena, // the dataspace
           const value_t * refStats, // the hyper-grid of reference tile variances
           const value_t * tgtStats, // the hyper-grid of target tile averages
           std::size_t rdim, std::size_t rcells, // ref grid shape and size
           std::size_t tdim, std::size_t tcells, // tgt grid shape and size
           std::size_t cdim, std::size_t ccells, // cor grid shape and size
           std::size_t row, std::size_t col,
           value_t * correlation)
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


    // N.B.: do not be tempted to terminate early threads that have no assigned workload; their
    // participation is required to make sure that shared memory is properly zeored out for the
    // nominally out of bounds accesses

    // get access to my shared memory
    extern __shared__ value_t scratch[];
    // get a handle to this thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // initialize the numerator term
    value_t numerator = 0;
    // initialize the target variance accumulator
    value_t tgtVariance = 0;
    // look up the mean target amplitude
    auto mean = tgtStats[b*ccells + row*cdim + col];

    // reference and target grids are interleaved; compute the stride
    std::size_t stride = rcells + tcells;

    // my {ref} starting point is column {t} of grid {b}
    auto ref = arena + b*stride + t;
    // my {tgt} starting point is column {t} of grid {b} at (row, col)
    // value_t * tgt = arena + b*stride + rcells + (row*tdim + col) + t;
    // or, more simply
    auto tgt = ref + rcells + (row*tdim + col);

    // if my thread id is less than the number of columns in the reference tile, i need to sum
    // up the contributions to the numerator and the target tile variance from my column; if
    // not, m y contribution is zero out my slots in shared memory
    if (t < rdim) {
        //run down the two columns
        for (std::size_t idx=0; idx < rdim; ++idx) {
            // fetch the ref value
            value_t r = ref[idx*rdim];
            // fetch the tgt value and subtract the mean target amplitude
            value_t t = tgt[idx*tdim] - mean;
            // update the numerator
            numerator += r * t;
            // and the target variance
            tgtVariance += t * t;
        }
    }

    // save my partial results
    scratch[2*t] = numerator;
    scratch[2*t + 1] = tgtVariance;
    // barrier: make sure everybody is done
    cta.sync();

    // now do the reduction in shared memory
    // for progressively smaller block sizes, the bottom half of the threads collect partial sums
    // N.B.: T is a template parameter, known at compile time, so it's easy for the optimizer to
    // eliminate the impossible clauses
    // for 512 threads per block
    if (T >= 512 && t < 256) {
        // my sibling's offset
        auto offset = 2*(t+256);
        // update my partial sum by reading my sibling's value
        numerator += scratch[offset];
        // ditto for the target variance
        tgtVariance += scratch[offset+1];
        // and make them available
        scratch[2*t] = numerator;
        scratch[2*t+1] = tgtVariance;
    }
    // make sure everybody is done
    cta.sync();

    // for 256 threads per block
    if (T >= 256 && t < 128) {
        // my sibling's offset
        auto offset = 2*(t+128);
        // update my partial sum by reading my sibling's value
        numerator += scratch[offset];
        // ditto for the target variance
        tgtVariance += scratch[offset+1];
        // and make them available
        scratch[2*t] = numerator;
        scratch[2*t+1] = tgtVariance;
    }
    // make sure everybody is done
    cta.sync();

    // for 128 threads per block
    if (T >= 128 && t < 64) {
        // my sibling's offset
        auto offset = 2*(t+64);
        // update my partial sum by reading my sibling's value
        numerator += scratch[offset];
        // ditto for the target variance
        tgtVariance += scratch[offset+1];
        // and make them available
        scratch[2*t] = numerator;
        scratch[2*t+1] = tgtVariance;
    }
    // make sure everybody is done
    cta.sync();

    // on recent architectures, there is a faster way to do the reduction once we reach the
    // warp level; the only cost is that we have to make sure there is enough memory for 64
    // threads, i.e. the shared memory size is bound from below by 64*sizeof(value_t)
    if (t < 32) {
        // if we need to
        if (T >= 64) {
            // my sibling's offset
            auto offset = 2*(t+32);
            // pull a neighbor's value
            numerator += scratch[offset];
            tgtVariance += scratch[offset+1];
        }
        // get a handle to the active thread group
        cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
        // the power-of-2 threads
        for (int offset = 16; offset > 0; offset >>= 1) {
            // reduce using {shuffle}
            numerator += active.shfl_down(numerator, offset);
            tgtVariance += active.shfl_down(tgtVariance, offset);
        }
    }

    // finally, the master thread of each block
    if (t == 0) {
        // looks up the sqrt of the reference tile variance
        value_t refVariance = refStats[b];
        // computes the correlation
        auto corr = numerator / refVariance / std::sqrt(tgtVariance);
        // computes the slot where this result goes
        std::size_t slot = b*ccells + row*cdim + col;
        // and writes the sum to the result vector
        correlation[slot] = corr;
    }

    // all done
    return;
}


// end of file
