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
        << "test: refining the coarse dataset"
        << pyre::journal::endl;

    // the reference tile extent
    int refDim = 64;
    // the margin around the reference tile
    int margin = 16;
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

    // the shape of the refined reference tiles
    auto refRefinedShape = refineFactor * refShape;
    // the shape of the refined target tiles
    auto tgtRefinedShape = refineFactor * (refShape + slc_t::index_type::fill(2*refineMargin));
    // the layout of the refined reference tiles
    slc_t::layout_type refRefinedLayout { refRefinedShape };
    // the layout of the refined target tiles
    slc_t::layout_type tgtRefinedLayout { tgtRefinedShape };
    // the number of cells in a refined reference tile
    auto refRefinedCells = refRefinedLayout.size();
    // the number of cells in a refined target tile
    auto tgtRefinedCells = tgtRefinedLayout.size();
    //  the number of cells per refined pair
    auto cellsPerRefinedPair = refRefinedCells + tgtRefinedCells;
    // the total number of refined cells
    auto cellsRefined = pairs * cellsPerRefinedPair;

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

    // we fill the reference tiles with random numbers
    // make a device
    std::random_device dev {};
    // a random number generator
    std::mt19937 rng { dev() };
    // use them to build a normal distribution
    std::normal_distribution<float> normal {};

    // start the clock
    timer.reset().start();
    // make a reference raster
    slc_t ref(refLayout);
    // and fill it
    for (auto idx : ref.layout()) {
        // with random numbers pulled from the normal distribution
        ref[idx] = normal(rng);
    }
    // make a view over the reference tile
    auto rview = ref.constview();
    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            auto slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            auto view = tgt.view(slice);
            // place a copy of the reference tile
            std::copy(rview.begin(), rview.end(), view.begin());

            // compute the pair id
            int pid = i*placements + j;
            // add this pair to the correlator
            c.addReferenceTile(pid, ref.constview());
            c.addTargetTile(pid, tgt.constview());
        }
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "creating reference dataset: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // push the data to the device
    auto coarseArena = c._push();
    // stop the clock
    timer.stop();
    // get the duration
    auto duration = timer.read();
    // get the payload
    auto footprint = cells * sizeof(slc_t::cell_type);
    // compute the transfer rate in Gb/s
    auto rate = footprint / duration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the dataset to the device: " << 1e3 * duration << " ms"
        << ", at " << rate << " Gb/s"
        << pyre::journal::endl;

    // start exercising the refinement process
    // start the clock
    timer.reset().start();
    // allocate a new arena
    auto refinedArena = c._refinedArena();
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "allocating the refined tile arena: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // refine the reference tiles
    c._refRefine(coarseArena, refinedArena);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "refining the reference tiles: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // let's bring the refined arena back to the host
    // the memory footprint
    auto refinedFootprint = cellsRefined * sizeof(slc_t::cell_type);
    // make room
    auto refined = new slc_t::cell_type[cellsRefined];
    // start the clock
    timer.reset().start();
    // copy the results over
    auto status = cudaMemcpy(refined, refinedArena, refinedFootprint, cudaMemcpyDeviceToHost);
    // stop the clock
    timer.stop();
    // if something went wrong
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while retrieving the refined tiles: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // get the duration
    duration = timer.read();
    // compute the transfer rate
    rate = refinedFootprint / duration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the refined tiles to the host: " << 1e3 * duration << " ms"
        << ", at " << rate << " Gb/s"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // establish a tolerance
    auto tolerance = std::numeric_limits<value_t>::epsilon();
    // verify that all the cells that would have been occupied by the target tiles are still zero
    for (auto pid=0; pid < pairs; ++pid) {
        // find the starting point of this target tile
        auto tgtTile = refined + pid*cellsPerRefinedPair + refRefinedCells;
        // make a tile over it
        tile_t tgt { tgtRefinedShape, tgtTile };
        // go through all the cells
        for (auto idx : tgt.layout()) {
            // get the value
            auto cell = tgt[idx];
            // check that it is zero
            if (std::abs(cell) > tolerance) {
                // make a channel
                pyre::journal::error_t error("ampcor.cuda");
                // show me
                error
                    << pyre::journal::at(__HERE__)
                    << "non-zero target cell: " << cell
                    << pyre::journal::newline
                    << "at [" << pid << ";" << idx << "]"
                    << pyre::journal::endl;
                // and bail
                throw std::runtime_error("verification error");
            }
        }
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "verifying that the refined target slots were untouched: " << 1e3 * duration << " ms"
        << pyre::journal::endl;

    // let's start over with the target refinement check
    // get rid of the refined arena
    cudaFree(refinedArena);
    // start the clock
    timer.reset().start();
    // allocate a new one
    refinedArena = c._refinedArena();
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "reallocating the refined tile arena: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // synthesize the locations of the maxima
    // start the clock
    timer.reset().start();
    // a (row, col) index for each pair
    int * maxloc = new int[2*pairs];
    // initialize it
    for (auto pid=0; pid < pairs; ++pid) {
        // decode the row and column and place the values
        maxloc[2*pid] = pid / placements;
        maxloc[2*pid + 1] = pid % placements;
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "synthesizing locations for the refined target tiles: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // on the device
    int * dloc;
    // we need
    auto locFootprint = 2 * pairs * sizeof(int);
    // make some room
    status = cudaMallocManaged(&dloc, locFootprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while allocating memory for the locations of the correlation maxima: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::bad_alloc();
    }
    // move the locations to the device
    status = cudaMemcpy(dloc, maxloc, locFootprint, cudaMemcpyHostToDevice);
    // if something went wrong
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while moving the locations of the correlation maxima to the device: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving maxima locations to the device: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // nudge them
    c._nudge(dloc, refDim, tgtDim);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "nudging locations: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // migrate
    c._tgtMigrate(coarseArena, dloc, refinedArena);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "migrating the expanded maxcor tiles to the refinement arena "
        << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;


    // start the clock
    timer.reset().start();
    // refine the target tiles
    c._tgtRefine(refinedArena);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "refining the target tiles: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // let's bring the refined arena back to the host
    // start the clock
    timer.reset().start();
    // copy the results over
    status = cudaMemcpy(refined, refinedArena, refinedFootprint, cudaMemcpyDeviceToHost);
    // stop the clock
    timer.stop();
    // if something went wrong
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while retrieving the refined tiles: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // get the duration
    duration = timer.read();
    // compute the transfer rate
    rate = refinedFootprint / duration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the second set of refined tiles to the host: " << 1e3 * duration << " ms"
        << ", at " << rate << " Gb/s"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // verify that all the cells that would have been occupied by the reference tiles are still zero
    for (auto pid=0; pid < pairs; ++pid) {
        // find the starting point of this target tile
        auto refTile = refined + pid*cellsPerRefinedPair;
        // make a tile over it
        tile_t ref { refRefinedShape, refTile };
        // go through all the cells
        for (auto idx : ref.layout()) {
            // get the value
            auto cell = ref[idx];
            // check that it is zero
            if (std::abs(cell) > tolerance) {
                // make a channel
                pyre::journal::error_t error("ampcor.cuda");
                // show me
                error
                    << pyre::journal::at(__HERE__)
                    << "non-zero reference cell: " << cell
                    << pyre::journal::newline
                    << "at [" << pid << ";" << idx << "]"
                    << pyre::journal::endl;
                // and bail
                throw std::runtime_error("verification error");
            }
        }
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "verifying that the refined reference slots were untouched: "
        << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // clean up
    delete [] maxloc;
    delete [] refined;
    cudaFree(dloc);
    cudaFree(refinedArena);
    cudaFree(coarseArena);

    // all done
    return 0;
}

// end of file
