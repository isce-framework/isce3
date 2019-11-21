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

// adapt a chunk of memory into a tile
using tile_t = pyre::grid::grid_t<slc_t::cell_type,
                                  slc_t::layout_type,
                                  pyre::memory::constview_t<value_t>>;

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
    int refDim = 32;
    // the margin around the reference tile
    int margin = 8;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;
    //  the dimension of the correlation matrix
    auto corDim = placements;

    // the number of pairs
    auto pairs = placements*placements;

    // the number of cells in a reference tile
    auto refCells = refDim * refDim;
    // the number of cells in a target tile
    auto tgtCells = tgtDim * tgtDim;
    // the number of cells in the table of mean values
    auto corCells = corDim * corDim;
    // the number of cells in each pair
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
    std::normal_distribution<float> normal {};

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
            auto slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            auto view = tgt.view(slice);
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
    // start the clock
    timer.reset().start();
    // compute the average amplitude of all possible ref shaped sub-tiles in the target tile
    auto tgtStats = c._tgtStats(sat, refDim, tgtDim, corDim);
    // stop the clock
    timer.stop();
    // get the duration
    duration = timer.read();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing averages: " << 1e3 * duration << " ms"
        << pyre::journal::endl;

    // verification: get results moved over to the host
    // verify: go through all the tables and verify that they contain the correct target means
    // we need the detected tiles, so make room for the results
    auto ampResults = new value_t[cells];
    // compute the result footprint
    auto ampFootprint = cells * sizeof(value_t);
    // start the clock
    timer.reset().start();
    // copy the results over
    cudaError_t status = cudaMemcpy(ampResults, rArena, ampFootprint, cudaMemcpyDeviceToHost);
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
            << "while retrieving the amplitudes of the tiles: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // get the duration
    auto ampDuration = timer.read();
    // compute the transfer rate
    auto ampRate = ampFootprint / ampDuration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the detected tiles to the host: " << 1e3 * ampDuration << " ms"
        << ", at " << ampRate << " Gb/s"
        << pyre::journal::endl;

    // we need the SATS
    auto satResults = new value_t[pairs * tgtCells];
    // compute the result footprint
    auto satFootprint = pairs * tgtCells * sizeof(value_t);
    // start the clock
    timer.reset().start();
    // copy the results over
    status = cudaMemcpy(satResults, sat, satFootprint, cudaMemcpyDeviceToHost);
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
            << "while retrieving SATs: "
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

    // finally, we need the hyper-grid with the average values
    // the total number of cells in the stats array: 2 floats per placement per pair
    auto statCells = pairs * corCells;
    // make room
    auto statResults = new value_t[statCells];
    // compute the memory footprint
    auto statFootprint = statCells * sizeof(value_t);
    // start the clock
    timer.reset().start();
    // copy the results over
    status = cudaMemcpy(statResults, tgtStats, statFootprint, cudaMemcpyDeviceToHost);
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
            << "while retrieving the hyper-grid of average amplitudes for the target tiles: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // get the duration
    auto statDuration = timer.read();
    // compute the transfer rate
    auto statRate = statFootprint / statDuration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the hyper-grid of averages to the host: " << 1e3 * statDuration << " ms"
        << ", at " << statRate << " Gb/s"
        << pyre::journal::endl;

    // compare expected with computed
    // set up a tolerance
    auto tolerance = 10 * std::numeric_limits<value_t>::epsilon();
    // start the clock
    timer.reset().start();
    // go through all the pairs
    for (auto pid = 0; pid < pairs; ++pid) {
        // find the beginning of the target tile
        value_t * tgtStart = ampResults + pid*cellsPerPair + refCells;
        // make a tile
        tile_t tgt { tgtLayout, tgtStart };
        // locate the table of mean values for this pair
        value_t * stats = statResults + pid*corCells;
        // go through all the placements
        for (auto i=0; i<corDim; ++i) {
            for (auto j=0; j<corDim; ++j) {
                // the offset to the stats for this tile for this placement
                auto offset = i*corDim + j;
                // slice the target tile
                auto slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
                // make a view
                auto view = tgt.constview(slice);
                // use it to compute the average value in the slice
                auto expectedMean = std::accumulate(view.begin(), view.end(), 0.0) / refCells;
                // read the computed value
                auto computedMean = stats[offset];
                // compute the mismatch
                auto mismatchMean = std::abs(1.0-expectedMean/computedMean);
                // verify it's near zero
                if (std::abs(mismatchMean) > tolerance) {
                    // make a channel
                    pyre::journal::error_t error("ampcor.cuda");
                    // complain
                    error
                        << pyre::journal::at(__HERE__)
                        << "mean mismatch at tile [" << pid << ":" << i << "," << j << "]: "
                        << "expected: " << expectedMean
                        << ", computed: " << computedMean
                        << pyre::journal::endl;
                    // bail
                    throw std::runtime_error("mean verification error!");
                }
            }
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
                    channel << satResults[pid*tgtCells + idx*tgtDim + jdx] << " ";
                }
                channel << pyre::journal::newline;
            }

            // the table of averages
            channel << "AVG:" << pyre::journal::newline;
            // find the AVG that corresponds to this pid and print it
            for (auto idx=0; idx < corDim; ++idx) {
                for (auto jdx=0; jdx < corDim; ++jdx) {
                    channel << statResults[pid*corCells + idx*corDim + jdx] << " ";
                }
                channel << pyre::journal::newline;
            }
        }
        channel << pyre::journal::endl;
    }

    // clean up
    cudaFree(tgtStats);
    cudaFree(sat);
    cudaFree(rArena);
    cudaFree(cArena);

    delete [] statResults;
    delete [] satResults;
    delete [] ampResults;

    // all done
    return 0;
}

// end of file
