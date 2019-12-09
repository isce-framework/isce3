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
// the correlation matrix
using cor_t = pyre::grid::simple_t<3, value_t>;
// the correlator
using correlator_t = ampcor::cuda::correlators::sequential_t<slc_t>;

// #define SHOW_ME

// driver
int main() {
    // pick a devvice
    cudaSetDevice(4);

    // number of gigabytes per byte
    const auto Mb = 1.0/(1024*1024);

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
    int refDim = 128;
    // the margin around the reference tile
    int margin = 32;
    // the refining margin
    int refineMargin = 8;
    // the refining factor
    int refineFactor = 2;
    // the zoom factor
    int zoomFactor = 4;

    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the shape of the base correlation matrix
    auto corDim = 2*refineFactor*refineMargin + 1;
    // the shape of the zoomed correlation matrix
    auto zmdDim = zoomFactor * corDim;

    // the number of pairs
    auto pairs = corDim * corDim;

    // the reference shape
    slc_t::shape_type refShape {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape {tgtDim, tgtDim};
    // the base correlation matrix shape
    cor_t::shape_type corShape {pairs, corDim, corDim};
    // the zoomed correlation matrix shape
    cor_t::shape_type zmdShape {pairs, zmdDim, zmdDim};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // the number of cells in a reference tile
    auto refCells = refShape.size();
    // the number of cells in a target tile
    auto tgtCells = tgtShape.size();
    // the number of cells in the correlation matrix
    auto corCells = corShape.size();
    // the number of cells in the zoomed correlation matrix
    auto zmdCells = zmdShape.size();

    // start the clock
    timer.reset().start();
    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout, refineFactor, refineMargin, zoomFactor);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "instantiating the manager: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // make a correlation matrix
    cor_t cor { corShape };
    // fill it
    for (auto idx : cor.layout()) {
        // extract the pair id
        auto pid = idx[0];
        // set the data
        cor[idx] = pid;
    }

#if defined(SHOW_ME)
    // show me
    for (auto pid = 0; pid < pairs; ++pid) {
        // tell me
        channel << "pair " << pid << ": " << pyre::journal::newline;
        // go through the correlation matrix
        for (auto idx = 0; idx < corDim; ++idx) {
            for (auto jdx = 0; jdx < corDim; ++jdx) {
                channel << cor[{pid, idx, jdx}] << " ";
            }
            channel << pyre::journal::newline;
        }
    }
    channel << pyre::journal::endl;
#endif

    // compute the footprint of the correlation matrix
    auto corFootprint = cor.layout().size() * sizeof(value_t);
    // grab a spot on the device
    value_t * gamma = nullptr;
    // allocate some memory
    auto status = cudaMallocManaged(&gamma, corFootprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while allocating memory for the correlation matrix: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::bad_alloc();
    }
    // move the correlation matrix to the device
    status = cudaMemcpy(gamma, cor.data(), corFootprint, cudaMemcpyHostToDevice);
    // if something went wrong
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while moving the correlation matrix to the device: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }
    // zoom
    auto zgamma = c._zoomcor(gamma);
    // make a grid for the zoomed correlation matrix
    cor_t zoomed {zmdShape};
    // compute its footprint
    auto zmdFootprint = zoomed.layout().size() * sizeof(value_t);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "moving " << zmdFootprint * Mb << " Mb from the device to host memory "
        << zoomed.data()
        << pyre::journal::endl;
    // move the device data into it
    status = cudaMemcpy(zoomed.data(), zgamma, zmdFootprint, cudaMemcpyDeviceToHost);
    // if something went wrong
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while moving the zoomed correlation matrix to the host: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        throw std::runtime_error(description);
    }

#if defined(SHOW_ME)
    // show me
    for (auto pid = 0; pid < pairs; ++pid) {
        // tell me
        channel << "pair " << pid << ": " << pyre::journal::newline;
        // go through the correlation matrix
        for (auto idx = 0; idx < zmdDim; ++idx) {
            for (auto jdx = 0; jdx < zmdDim; ++jdx) {
                channel << zoomed[{pid, idx, jdx}] << " ";
            }
            channel << pyre::journal::newline;
        }
    }
    channel << pyre::journal::endl;
#endif


    // clean up
    cudaFree(zgamma);
    cudaFree(gamma);

    // all done
    return 0;
}

// end of file
