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
_detect(const cuFloatComplex * cArena, std::size_t cells, std::size_t load, float * rArena);


// compute the amplitude of the signal tiles, assuming pixels are of type std::complex<float>
void
ampcor::cuda::kernels::
detect(const std::complex<float> * cArena, std::size_t cells, float * rArena)
{
    // this is embarrassingly parallel, so pick a simple deployment schedule
    // the load of each thread
    std::size_t N = 128*128;
    // the number of threads per block
    std::size_t T = 128;
    // the number of cells handled by a block
    std::size_t L = N*T;
    // hence, the number of blocks
    std::size_t B = (cells / L) + (cells % L ? 1 : 0);

    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each to process "
        << cells << " cells"
        << pyre::journal::endl;

    // launch
    _detect <<<B,T>>> (reinterpret_cast<const cuFloatComplex *>(cArena), cells, N, rArena);
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
            << "while computing pixel amplitudes: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// implementations
__global__
static void
_detect(const cuFloatComplex * cArena, std::size_t cells, std::size_t load, float * rArena)
{
    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;      // number of blocks
    std::size_t T = blockDim.x;        // number of threads per block
    // auto W = B*T;                   // total number of workers
    // local
    std::size_t b = blockIdx.x;        // my block id
    std::size_t t = threadIdx.x;       // my thread id
    // auto w = b*T + t;               // my worker id

    // the number of cells handled by each block
    auto L = T * load;
    // the number of cells handled by the blocks before me
    auto skip = b*L;
    // threads in this block should go no further than
    auto stop = min((b+1)*L, cells);

#if defined(DEBUG_DETECT)
    // the first thread of each block
    if (t == 0) {
        // show me
        printf("[%05lu]: skip=%lu, stop=%lu\n", w, skip, stop);
    }
#endif

    // go through my cells
    for (auto current=skip+t; current < stop; current += T) {
        // get the complex pixel, compute its amplitude and store it
        rArena[current] = cuCabsf(cArena[current]);
    }

    // all done
    return;
}


// end of file
