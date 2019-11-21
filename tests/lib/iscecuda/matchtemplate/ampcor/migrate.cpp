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

// complex literals
using namespace std::literals::complex_literals;

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
        ref[idx] = 1; // normal(rng) + 1if*normal(rng);
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

    // synthesize the locations of the maxima
    // start the clock
    timer.reset().start();
    // we build a (row, col) index for each pair
    auto locCells = 2 * pairs;
    // which occupies
    auto locFootprint = locCells * sizeof(int);
    // allocate
    int * loc = new int[locCells];
    // initialize it
    for (auto pid=0; pid < pairs; ++pid) {
        // decode the row and column and place the values
        loc[2*pid] = pid / placements;
        loc[2*pid + 1] = pid % placements;
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
    // make some room
    auto status = cudaMallocManaged(&dloc, locFootprint);
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
    status = cudaMemcpy(dloc, loc, locFootprint, cudaMemcpyHostToDevice);
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

    // reset the clock
    timer.reset().start();
    // make some room
    auto refined = new slc_t::cell_type[cellsRefined];
    // with footprint
    footprint = cellsRefined * sizeof(slc_t::cell_type);
    // move the data from the device
    status = cudaMemcpy(refined, refinedArena, footprint, cudaMemcpyDeviceToHost);
    // if something went wrong
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while harvesting the refined tiles: "
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
        << "harvesting the refined tiles " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // reset the clock
    timer.reset().start();
    // move the nudged locations from the device
    status = cudaMemcpy(loc, dloc, locFootprint, cudaMemcpyDeviceToHost);
    // if something went wrong
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while harvesting the nudged locations: "
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
        << "harvesting the nudged locations " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // verify
    // establish a tolerance
    auto tolerance = std::numeric_limits<value_t>::epsilon();
    // the dimension of the expanded maxcor slice
    auto expDim = refDim + 2*refineMargin;
    // reset the clock
    timer.reset().start();
    // go through all the pairs
    for (auto pid = 0; pid < pairs; ++pid) {
        // decode the pair id into an index
        auto row = pid / placements;
        auto col = pid % placements;
        // compute the beginning of the target tile in the refined arena
        auto trmem = refined + pid*cellsPerRefinedPair + refRefinedCells;
        // build a grid over it
        tile_t tgtRefined { tgtRefinedLayout, trmem };

#if defined(SHOW_TILE)
        // show me the tile
        channel
            << pyre::journal::at(__HERE__)
            << "pair (" << row << "," << col << "): " << pyre::journal::newline;
        for (auto idx : tgtRefined.layout()) {
            if (idx[1] == 0) channel << pyre::journal::newline;
            channel << tgtRefined[idx];
        }
        channel << pyre::journal::endl;
#endif

        // compute the beginning of the correct target tile
        auto tmem = c.arena() + pid*cellsPerPair + refCells;
        // build a grid over it
        tile_t tgt { tgtLayout, tmem };
        // find the ULHC of the tile expanded maxcor tile in the target tile
        auto base = tile_t::index_type {loc[2*pid], loc[2*pid+1]};

        // go through it
        for (auto idx : tgtRefined.layout()) {
            // in the expanded region
            if (idx[0] >= expDim || idx[1] >= expDim) {
                // make sure we have a zero
                if (std::abs(tgtRefined[idx]) > tolerance) {
                    // make a channel
                    pyre::journal::error_t error("ampcor.cuda");
                    // complain
                    error
                        << pyre::journal::at(__HERE__)
                        << "pair " << pid << ": mismatch at (" << idx << "): "
                        << "expected zero, got " << tgtRefined[idx]
                        << pyre::journal::endl;
                    // and bail
                    throw std::runtime_error("verification error!");
                }
            } else {
                // what i got
                auto actual = tgtRefined[idx];
                // what i expect
                auto expected = tgt[base+idx];
                // if there is a mismatch
                if (actual != expected) {
                    // make a channel
                    pyre::journal::error_t error("ampcor.cuda");
                    // complain
                    error
                        << pyre::journal::at(__HERE__)
                        << "pair (" << pid/placements << "," << pid%placements << ")"
                        ": mismatch at (" << idx << "): "
                        << "expected: " << expected
                        << ", got: " << actual << " from tgt[" << base+idx << "]"
                        << ", expDim: " << expDim
                        << pyre::journal::endl;
                    // and bail
                    throw std::runtime_error("verification error!");
                }
            }
        }
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "verifying that the target tiles were migrated correctly "
        << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // clean up
    delete [] refined;
    delete [] loc;
    cudaFree(dloc);
    cudaFree(refinedArena);
    cudaFree(coarseArena);

    // all done
    return 0;
}

// end of file
