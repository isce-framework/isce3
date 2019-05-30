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
        << "test: adjusting the amplitudes of the reference tile to zero mean"
        << pyre::journal::endl;

    // the reference tile extent
    int refDim = 64;
    // the margin around the reference tile
    int margin = 16;
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
    // the number of cells in each pair
    auto cellsPerPair = refDim*refDim + tgtDim*tgtDim;
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
            std::copy(ref.view().begin(), ref.view().end(), view.begin());

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
        << "moving the dataset to the device: " << 1e3 * wDuration << " ms"
        << ", at " << wRate << " Gb/s"
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
    // subtract the tile mean from each pixel in the reference tile and compute the variance
    auto refStats = c._refStats(rArena, refDim, tgtDim);
    // stop the clock
    timer.stop();
    // get the duration
    duration = timer.read();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing the statistics of the reference tiles: " << 1e3 * duration << " ms"
        << pyre::journal::endl;

    // make room for the detected tiles
    auto * tiles = new value_t[cells];
    // compute the result footprint
    auto rFootprint = cells * sizeof(value_t);
    // start the clock
    timer.reset().start();
    // copy the results over
    cudaError_t status = cudaMemcpy(tiles, rArena, rFootprint, cudaMemcpyDeviceToHost);
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
            << "while retrieving the detected tiles: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // get the duration
    auto rDuration = timer.read();
    // compute the transfer rate
    auto rRate = rFootprint / rDuration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the detected tiles to the host: " << 1e3 * rDuration << " ms"
        << ", at " << rRate << " Gb/s"
        << pyre::journal::endl;

    // make room for the reference tile statistics
    auto * stats = new value_t[pairs];
    // compute the result footprint
    auto sFootprint = pairs * sizeof(value_t);
    // start the clock
    timer.reset().start();
    // copy the results over
    status = cudaMemcpy(stats, refStats, sFootprint, cudaMemcpyDeviceToHost);
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
            << "while computing the statistics of the reference tiles: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // get the duration
    auto sDuration = timer.read();
    // compute the transfer rate
    auto sRate = sFootprint / sDuration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving reference tile statistics to the host: " << 1e3 * sDuration << " ms"
        << ", at " << sRate << " Gb/s"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // establish a tolerance
    auto tolerance = 10 * std::numeric_limits<value_t>::epsilon();
    // make a lambda that accumulates the square of a value
    auto square = [] (value_t partial, value_t pxl) -> value_t { return partial + pxl*pxl; };
    // verify
    for (auto pid = 0; pid < pairs; ++pid) {
        // compute the starting address of this tile
        auto tile = tiles + pid * cellsPerPair;
        // compute the mean of the tile
        auto mean = std::accumulate(tile, tile+refCells, 0.0) / refCells;
        // verify it's near zero
        if (std::abs(mean) > tolerance) {
            // make a channel
            pyre::journal::error_t error("ampcor.cuda");
            // complain
            error
                << pyre::journal::at(__HERE__)
                << "mean mismatch at tile " << pid << ": " << mean << " != 0"
                << pyre::journal::endl;
            // bail
            throw std::runtime_error("mean verification error!");
        }
        // compute the variance of the tile
        auto expectedVar = std::sqrt(std::accumulate(tile, tile+refCells, 0.0, square));
        // the computed value
        auto computedVar = refStats[pid];
        // compute the mismatch
        auto mismatchVar = std::abs(1 - computedVar/expectedVar);
        // show me
        channel
            << pyre::journal::at(__HERE__)
            << "tile " << pid << ": "
            << "expected variance: " << expectedVar
            << ", computed variance: " << computedVar
            << pyre::journal::endl;

        // if there is serious mismatch
        if (mismatchVar > 10*tolerance) {
            // make a channel
            pyre::journal::error_t error("ampcor.cuda");
            // complain
            error
                << pyre::journal::at(__HERE__)
                << "variance mismatch at tile " << pid << ": "
                << "computed: " << computedVar
                << ", expected: " << expectedVar
                << pyre::journal::endl;
            // bail
            throw std::runtime_error("variance verification error!");
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

            channel << "zero-mean reference:" << pyre::journal::newline;
            // the amplitude of the reference tile
            for (auto idx=0; idx < refDim; ++idx) {
                for (auto jdx=0; jdx < refDim; ++jdx) {
                    channel << tiles[pid*cellsPerPair + idx*refDim + jdx] << " ";
                }
                channel << pyre::journal::newline;
            }
        }
        channel << pyre::journal::endl;
    }

    // clean up
    cudaFree(refStats);
    cudaFree(cArena);
    cudaFree(rArena);

    delete [] stats;
    delete [] tiles;

    // all done
    return 0;
}

// end of file
