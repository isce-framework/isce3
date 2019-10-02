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
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"

// helpers
static void
_c2r(const std::complex<float> * scratch, std::size_t pairId, std::size_t zmdDim, float * zoomed);


// compute the amplitude of the signal tiles, assuming pixels are of type std::complex<float>
auto
ampcor::kernels::
c2r(const std::complex<float> * scratch, std::size_t pairs, std::size_t zmdDim) -> float *
{
    // grab a spot
    float * zoomed = nullptr;
    // compute the number of cells
    auto zmdCells = zmdDim * zmdDim;
    // allocate memory for the hyper-matrix
    zoomed = new (std::nothrow) float[pairs*zmdCells]();
    // if something went wrong
    if (zoomed == nullptr) {
        // make a channel
        pyre::journal::error_t error("ampcor");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while allocating memory for the new real-valued array"
            << pyre::journal::endl;
        // and bail
        throw std::bad_alloc();
    }

    #pragma omp parallel for
    for (std::size_t pairId=0; pairId <pairs; pairId++)
        _c2r(scratch, pairId, zmdDim, zoomed);

    // all done
    return zoomed;
}


// implementations
static void
_c2r(const std::complex<float> * scratch, std::size_t pairId, std::size_t zmdDim, float * zoomed)
{

    // compute the number of cells in a zoomed correlation matrix
    auto zmdCells = zmdDim * zmdDim;

    // find the matrix I'm reading from
    auto src = scratch + pairId * zmdCells;

    // and the matrix I'm writing to
    auto dst = zoomed + pairId * zmdCells;

    // transfer the whole {gamma} to {scratch}
    for (std::size_t idx = 0; idx < zmdCells; idx++) {
        // read the data, convert to complex, and store
        *dst = std::abs(*src);
        // update the pointers:
        src++;
        dst++;
    }

    // all done
    return;
}


// end of file
