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
#include <numeric>
#include <random>
// cuda
#include <cuda_runtime.h>
#include <cufft.h>
// support
#include <pyre/grid.h>
#include <pyre/journal.h>
#include <pyre/timers.h>
// ampcor
#include <isce/cuda/matchtemplate/ampcor/public.h>

// type aliases
// my value type
using value_t = float;
// the pixel type
using pixel_t = std::complex<value_t>;
// my raster type
using slc_t = pyre::grid::simple_t<2, pixel_t>;
// the correlator
using correlator_t = ampcor::cuda::correlators::sequential_t<slc_t>;

// adapt a chunk of memory into a tile
using mem_t = pyre::memory::constview_t<slc_t::cell_type>;
using tile_t = pyre::grid::grid_t<slc_t::cell_type, slc_t::layout_type, mem_t>;

// driver
int main() {
    // number of gigabytes per byte
    const auto Gb = 1.0/(1024*1024*1024);

    // make a timer
    pyre::timer_t timer("ampcor.cuda.sanity");
    // make a channel for reporting the timings
    pyre::journal::debug_t tlog("ampcor.cuda.tlog");

    // make a channel for logging progress
    pyre::journal::debug_t channel("ampcor.cuda");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "test: adjusting the locations of the refined target tiles"
        << pyre::journal::endl;

    // the reference tile extent
    int refDim = 128;
    // the margin around the reference tile
    int margin = 32;
    // the refinement factor
    int refineFactor = 2;
    // the margin around the refined target tile
    int refineMargin = 8;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;
    // the number of pairs
    auto pairs = placements*placements;

    // the number of cells in a reference tile
    auto refCells = refDim * refDim;
    // the number of cells in a target tile
    auto tgtCells = tgtDim * tgtDim;
    // the number of cells per pair
    auto cellsPerPair = refCells + tgtCells;
    // the total number of cells
    auto cells = pairs * cellsPerPair;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};
    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // start the clock
    timer.reset().start();
    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout, refineFactor, refineMargin);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "instantiating the manager: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // make an array of locations to simulate the result of {_maxcor}
    int * loc = new int[2*pairs];
    // fill with out standard strategy: the (r,c) target tile have a maximum at (r,c)
    for (auto pid = 0; pid < pairs; ++pid) {
        // decode the row and column
        int row = pid / placements;
        int col = pid % placements;
        // record
        loc[2*pid] = row;
        loc[2*pid + 1] = col;
    }

    // start the clock
    timer.reset().start();
    // on the device
    int * dloc;
    // the footprint
    auto footprint = 2 * pairs * sizeof(int);
    // allocate room
    auto status = cudaMallocManaged(&dloc, footprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while allocating " << 1.0*footprint/1024/1024
            << "Mb of device memory for the maxima locations: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::bad_alloc();
    }
    // move the data
    status = cudaMemcpy(dloc, loc, footprint, cudaMemcpyHostToDevice);
    // if something went wrong
    if (status != cudaSuccess) {
        // build the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while transferring locations to the device: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "synthesizing the dataset: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // nudge
    c._nudge(dloc, refDim, tgtDim);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "nudging the locations: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // bring back the data
    status = cudaMemcpy(loc, dloc, footprint, cudaMemcpyDeviceToHost);
    // if something went wrong
    if (status != cudaSuccess) {
        // build the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while harvesting the nudged locations from the device: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "harvesting the nudged locations: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // the lowest possible index
    int low = 0;
    // and the highest possible index
    int high = tgtDim - (refDim + 2*refineMargin);
    // go through the locations
    for (auto pid = 0; pid < pairs; ++pid) {
        // get the row and column
        auto row = loc[2*pid];
        auto col = loc[2*pid + 1];
        // verify
        if ((row < low) || (col < low) || (row > high) || (col > high)) {
            // decode the pair id
            int r = pid / placements;
            int c = pid % placements;
            // make a channel
            pyre::journal::error_t error("ampcor.cuda");
            // complain
            error
                << pyre::journal::at(__HERE__)
                << "pair (" << r << "," << c << "): bad location: (" << row << "," << col << ")"
                << ", mimimum: (" << low << "," << low << ")"
                << ", maximum: (" << high << "," << high << ")"
                << pyre::journal::endl;
            // bail
            throw std::runtime_error("verification error!");
        }
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "verifying the nudged locations: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // clean up
    delete [] loc;
    cudaFree(dloc);

    // all done
    return 0;
}

// end of file
