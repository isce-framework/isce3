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


// the kernel that assembles the offset field
template <typename value_t = float>
__global__
static void
_offsetField(const int * coarse,       // the coarse offsets
             const int * fine,         // the fine offsets
             std::size_t pairs,        // the total number of tiles
             std::size_t margin,       // the origin of the coarse shifts
             std::size_t refineMargin, // origin of the refined shifts
             std::size_t zoom,         // the overall zoom factor of the refined shifts
             float * field             // results
             );


// run through the correlation matrix for each, find its maximum value and record its location
void
ampcor::cuda::kernels::
offsetField(const int * coarse,       // the coarse offsets
            const int * fine,         // the fine offsets
            std::size_t pairs,        // the total number of entries
            std::size_t margin,       // the origin of the coarse shifts
            std::size_t refineMargin, // origin of the refined shifts
            std::size_t zoom,         // the overall zoom factor of the refined shifts
            float * field             // results
            )
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
        << " threads each to assemble the offset fields of " << pairs << " tiles"
        << pyre::journal::endl;
    // launch
    _offsetField <<<B,T>>> (coarse, fine, pairs, margin, refineMargin, zoom, field);
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
            << "while assembling the offset field: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// the kernel that assembles the offset field
template <typename value_t>
__global__
static void
_offsetField(const int * coarse,       // the coarse offsets
             const int * fine,         // the fine offsets
             std::size_t pairs,        // the total number of tiles
             std::size_t margin,       // the origin of the coarse shifts
             std::size_t refineMargin, // origin of the refined shifts
             std::size_t zoom,         // the overall zoom factor of the refined shifts
             float * field             // results
             )
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

    // a constant
    const value_t one = 1.0;
    // find the beginning of my coarse offset
    auto myCoarse = coarse + 2*w;
    // find the beginning of my fine offset
    auto myFine = fine + 2*w;
    // and the beginning of where i store my result
    auto myField = field + 2*w;

    // do the math
    myField[0] = (one*myCoarse[0] - margin) + (one * myFine[0] / zoom - refineMargin);
    myField[1] = (one*myCoarse[1] - margin) + (one * myFine[1] / zoom - refineMargin);

    // all done
    return;
}


// end of file
