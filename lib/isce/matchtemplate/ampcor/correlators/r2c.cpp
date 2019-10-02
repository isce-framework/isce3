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
_r2c(const float * gamma, std::size_t pairId, std::size_t corDim, std::size_t zmdDim, std::complex<float> * scratch);


// convert signal tiles which are assumed to be float to std::complex<float>
auto
ampcor::kernels::
r2c(const float * gamma,
    std::size_t pairs,
    std::size_t corDim, std::size_t zmdDim) -> std::complex<float> *
{

    // grab a spot
    std::complex<float> * scratch = nullptr;

    // compute the number of cells in the zoomed correlation matrix
    auto zmdCells = zmdDim * zmdDim;

    // allocate memory for the complex zoomed version
    scratch = new (std::nothrow) std::complex<float>[pairs*zmdCells]();

    // if something went wrong
    if (scratch == nullptr) {
        // make a channel
        pyre::journal::error_t error("ampcor");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while allocating memory for the new complex array"
            << pyre::journal::endl;
        // and bail
        throw std::bad_alloc();
    }
    // initialize the memory

    // launch
    #pragma omp parallel for
    for (std::size_t pairId=0; pairId < pairs; pairId++)
       _r2c(gamma, pairId, corDim, zmdDim, scratch);


    // all done
    return scratch;
}


// implementations
static void
_r2c(const float * gamma, std::size_t pairId, std::size_t corDim, std::size_t zmdDim, std::complex<float> * scratch)
{

    // compute the number of cells in the correlation matrix
    auto corCells = corDim * corDim;
    // compute the number of cells in a zoomed correlation matrix
    auto zmdCells = zmdDim * zmdDim;

    // find the matrix I'm reading from 
    auto src = gamma + pairId * corCells;
    // and the matrix I'm writing to 
    auto dst = scratch + pairId * zmdCells;


    // transfer the current {gamma} to {scratch}
    for (std::size_t idy = 0; idy < corDim; idy++) {

       // skip to current line of {scratch}. 
       // Current line of {gamma} is contiguous
       dst = scratch + pairId * zmdCells + idy*zmdDim; 

       for (std::size_t idx = 0; idx < corDim; idx++) {
          // read the data, convert to complex, and store
          *dst = {*src, 0.0};
          // update the pointers:
          src++;
          dst++;
       }
    }

    // all done
    return;
}


// end of file
