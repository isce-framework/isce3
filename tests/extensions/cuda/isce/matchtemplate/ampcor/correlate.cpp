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

// adapt a chunk of memory into a correlation hyper-matrix
using crep_t = std::array<int, 4>;
using cindex_t = pyre::grid::index_t<crep_t>;
using clayout_t = pyre::grid::layout_t<cindex_t>;
using cmem_t = pyre::memory::constview_t<value_t>;
using ctile_t = pyre::grid::grid_t<value_t, clayout_t, cmem_t>;

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
    int refDim = 64;
    // the margin around the reference tile
    int margin = 16;
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
        // with random numbers pulled from the normal distribution
        ref[idx] = normal(rng);
    }
    // make a view over the reference tile
    auto rview = ref.constview();
    // build the target tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            auto slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            auto tgtView = tgt.view(slice);
            // fill it with the contents of the reference tile for this pair
            std::copy(rview.begin(), rview.end(), tgtView.begin());

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
    auto duration = timer.read();
    // get the payload
    auto footprint = cells * sizeof(slc_t::cell_type);
    // compute the transfer rate in Gb/s
    auto rate = footprint / duration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the dataset to the device: " << 1e3 * duration << " ms,"
        << " at " << rate << " Gb/s"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // compute the amplitude of every pixel
    auto rArena = c._detect(cArena, refDim, tgtDim);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing amplitudes of the signal tiles: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // compute reference tile statistics
    auto refStats = c._refStats(rArena, refDim, tgtDim);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing reference tile statistics: " << 1e3 * timer.read() << " ms,"
        << " at " << rate << " Gb/s"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // compute the sum area tables
    auto sat = c._sat(rArena, refDim, tgtDim);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing sum area tables: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // compute the average amplitude of all possible ref shaped sub-tiles in the target tile
    auto tgtStats = c._tgtStats(sat, refDim, tgtDim, corDim);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing target tile statistics: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // compute the average amplitude of all possible ref shaped sub-tiles in the target tile
    auto gamma = c._correlate(rArena, refStats, tgtStats, refDim, tgtDim, corDim);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing the correlation matrix: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // fetch the correlation matrix
    // make room for the correlation hyper-matrix
    auto cor = new value_t[pairs * corCells];
    // compute how much memory it occupies
    footprint = pairs * corCells * sizeof(value_t);
    // start the clock
    timer.reset().start();
    // copy the results over
    cudaError_t status = cudaMemcpy(cor, gamma, footprint, cudaMemcpyDeviceToHost);
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
            << "while retrieving the correlation hyper-matrix: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }
    // get the duration
    duration = timer.read();
    // compute the transfer rate
    rate = footprint / duration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the correlation hyper-matrix to the host: " << 1e3 * duration << " ms"
        << ", at " << rate << " Gb/s"
        << pyre::journal::endl;

    // the shape of the correlation matrix: the first two indices identify the pair, the last
    // two indicate the placement of the reference tile within the target search window
    ctile_t::shape_type corShape = {corDim, corDim, corDim, corDim};
    // the layout of the correlation matrix
    ctile_t::layout_type corLayout = { corShape };
    // adapt the correlation matrix into a grid
    ctile_t cgrid { corLayout, cor };

    // establish a tolerance
    auto tolerance = 10 * std::numeric_limits<value_t>::epsilon();
    // verify by checking that the correlation is unity for the correct placement of the
    // reference tile within the target window
    for (auto idx = 0; idx < corDim; ++idx) {
        for (auto jdx = 0; jdx < corDim; ++jdx) {
            // we expect
            auto expectedCor = 1.0f;
            // the magic placement should also have unit correlation
            auto computedCor = cgrid[{idx, jdx, idx, jdx}];
            // compute the mismatch
            auto mismatch = std::abs(1 - computedCor/expectedCor);
            // if there is significant mismatch
            if (mismatch > tolerance) {
                // make a channel
                pyre::journal::error_t error("ampcor.cuda");
                // show me
                error
                    << pyre::journal::at(__HERE__)
                    << "correlation mismatch at {" << idx << ", " << jdx << "}: "
                    << " expected: " << expectedCor
                    << ", computed: " << computedCor
                    << pyre::journal::endl;
                // and bail
                throw std::runtime_error("verification error");
            }
        }
    }

    // clean up
    delete [] cor;

    cudaFree(gamma);
    cudaFree(tgtStats);
    cudaFree(sat);
    cudaFree(rArena);
    cudaFree(cArena);

    // all done
    return 0;
}

// end of file
