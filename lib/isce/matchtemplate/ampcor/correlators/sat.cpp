// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>
// STL
#include <complex>
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"


// the SAT generation kernel
template <typename value_t = float>
static void
_sat(const value_t * dArena,
    std::size_t pairId, std::size_t rcells, std::size_t tcells, std::size_t tdim,
    value_t * dSAT);


// implementation
void
ampcor::kernels::
sat(const float * dArena,
    std::size_t pairs, std::size_t refCells, std::size_t tgtCells, std::size_t tgtDim,
    float * dSAT)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor");

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching SATs computation"
        << pyre::journal::endl;


    // launch the SAT kernel
    #pragma omp parallel for
    for (std::size_t pairId = 0; pairId < pairs; pairId++)
        _sat(dArena, pairId, refCells, tgtCells, tgtDim, dSAT);

    // all done
    return;
}


// the SAT generation kernel
template <typename value_t>
void
_sat(const value_t * dArena,
    std::size_t pairId, std::size_t rcells, std::size_t tcells, std::size_t tdim,
    value_t * dSAT)
{
    // Get the stride from one pair to the other
    std::size_t stride = rcells + tcells;
 
    
    // my starting point for reading data in the arena
    std::size_t read = pairId*stride + rcells;
    // my starting point for writing data in the SAT area
    std::size_t write = pairId*tcells;


    // First pixel
    dSAT[write] = dArena[read];

    // First row
    for (std::size_t col=1; col < tdim; col++)
       dSAT[write+col] = dSAT[write+col-1] + dArena[read+col];

    // Next rows
    for (std::size_t row=1; row < tdim; row++) {

        std::size_t offsetWrite1 = write + tdim * row;
        std::size_t offsetWrite2 = write + tdim * (row - 1);
        std::size_t offsetRead1  = read  + tdim * row;

        // First pixel of the current row
        // current row cumulative sum
        value_t sum = dArena[offsetRead1];
        dSAT[offsetWrite1] = dSAT[offsetWrite2] + sum;

        // Next pixels
        for (std::size_t col=1; col < tdim; col++) {
           sum += dArena[offsetRead1 + col];
           dSAT[offsetWrite1 + col] = dSAT[offsetWrite2 + col] + sum;
        }
    } 

    // all done
    return;
}


// end of file
