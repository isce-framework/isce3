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
// my pixel type
using pixel_t = std::complex<value_t>;
// my raster type
using slc_t = pyre::grid::simple_t<2, pixel_t>;
// the correlator
using correlator_t = ampcor::cuda::correlators::sequential_t<slc_t>;

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
        << "setting up the correlation plan with the cuda ampcor task manager"
        << pyre::journal::endl;

    // the reference tile extent
    int refDim = 2;
    // the margin around the reference tile
    int margin = 1;
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
    correlator_t c(pairs, refLayout, tgtLayout);
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
    std::normal_distribution<value_t> normal {};

    // start the clock
    timer.reset().start();
    // make a reference raster
    slc_t ref(refLayout);
    // and fill it
    for (auto idx : ref.layout()) {
        // with random number pulled from the normal distribution
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
            slc_t::slice_type slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            slc_t::view_type view = tgt.view(slice);
            // fill it with the contents of the reference tile for this pair
            std::copy(rview.begin(), rview.end(), view.begin());
            // compute the pair id
            int pid = i*placements + j;
            // add this pair to the correlator
            c.addReferenceTile(pid, rview);
            c.addTargetTile(pid, tgt.constview());
        }
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "creating synthetic dataset: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // push the data to the device
    auto cArena = c._push();
    // stop the clock
    timer.stop();
    // get the duration
    auto wDuration = timer.read();
    // get the payload
    auto wFootprint = cells * sizeof(slc_t::cell_type);
    // compute the transfer rate in Gb/s
    auto wRate = wFootprint / wDuration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the dataset to the device: " << 1e3 * wDuration << " ms,"
        << " at " << wRate << " Gb/s"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // compute the amplitude of every pixel
    auto rArena = c._detect(cArena, refDim, tgtDim);
    // stop the clock
    timer.stop();
    // get the duration
    auto duration = timer.read();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing amplitudes of the signal tiles: " << 1e3 * duration << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // compute the sum area tables
    auto sat = c._sat(rArena, refDim, tgtDim);
    // stop the clock
    timer.stop();
    // get the duration
    duration = timer.read();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing sum area tables: " << 1e3 * duration << " ms"
        << pyre::journal::endl;

    // make room for the results
    auto satResults = new value_t[pairs * tgtCells];
    // compute the result footprint
    auto satFootprint = pairs * tgtCells * sizeof(value_t);
    // start the clock
    timer.reset().start();
    // copy the results over
    cudaError_t status = cudaMemcpy(satResults, sat, satFootprint, cudaMemcpyDeviceToHost);
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
            << "while retrieving the results: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // get the duration
    auto satDuration = timer.read();
    // compute the transfer rate
    auto satRate = satFootprint / satDuration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving SATs to the host: " << 1e3 * satDuration << " ms"
        << ", at " << satRate << " Gb/s"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // verify: go through all the tables and check that the lower right hand corner contains
    // the sum of all the tile elements
    for (auto pid = 0; pid < pairs; ++pid) {
        // compute the start of this tile
        auto begin = rArena + pid*cellsPerPair + refDim*refDim;
        // and one past the end of this tile
        auto end = begin + tgtDim*tgtDim;
        // compute the sum
        auto expected = std::accumulate(begin, end, 0.0);
        // get the value form the LRC of the corresponding SAT
        auto computed = satResults[(pid+1)*tgtDim*tgtDim - 1];
        // compute the difference
        auto mismatch = std::abs(1.0-computed/expected);
        // if the two don't match within 10 float epsilon
        if (mismatch > 10*std::numeric_limits<value_t>::epsilon()) {
            // make a channel
            pyre::journal::error_t error("ampcor.cuda");
            // complain
            error
                << pyre::journal::at(__HERE__)
                << "mismatch in SAT[" << pid << "]: "
                << "expected: " << expected
                << ", found: " << computed
                << ", mismatch: " << mismatch
                << pyre::journal::endl;
            // and bail
            throw std::runtime_error("verification error!");
        }
    }
    // stop the clock
    timer.stop();
    // get the duration
    auto vDuration = timer.read();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "verifying results at the host: " << 1e3 * vDuration << " ms"
        << pyre::journal::endl;

    // if the debug channel is active
    if (channel) {
        // dump the resulting pairs
        for (auto pid = 0; pid < pairs; ++pid) {
            // sign in
            channel
                << pyre::journal::at(__HERE__)
                << "--------------------"
                << pyre::journal::newline
                << "pair " << pid << ":"
                << pyre::journal::newline;

            // the target tile
            channel << "TGT:" << pyre::journal::newline;
            // find the tile that corresponds to this pid and print it
            for (auto idx=0; idx < tgtDim; ++idx) {
                for (auto jdx=0; jdx < tgtDim; ++jdx) {
                    channel << rArena[pid*cellsPerPair + refDim*refDim + idx*tgtDim + jdx] << " ";
                }
                channel << pyre::journal::newline;
            }

            // the SAT
            channel << "SAT:" << pyre::journal::newline;
            // find the SAT that corresponds to this pid and print it
            for (auto idx=0; idx < tgtDim; ++idx) {
                for (auto jdx=0; jdx < tgtDim; ++jdx) {
                    channel << satResults[pid*tgtDim*tgtDim + idx*tgtDim + jdx] << " ";
                }
                channel << pyre::journal::newline;
            }
        }
        channel << pyre::journal::endl;
    }

    // clean up
    cudaFree(sat);
    cudaFree(rArena);
    cudaFree(cArena);
    delete [] satResults;

    // all done
    return 0;
}

// end of file
