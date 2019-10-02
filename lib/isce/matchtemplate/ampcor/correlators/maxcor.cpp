// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// configuration
//#include <portinfo>
// pyre
#include <pyre/journal.h>
// pull the declarations
#include "kernels.h"

// the correlation maximum location kernel
template <typename value_t = float>
static void
_maxcor(const value_t * gamma,
        std::size_t pairs, std::size_t corCells, std::size_t corDim,
        int * loc);


// run through the correlation matrix for each, find its maximum value and record its location
void
ampcor::kernels::
maxcor(const float * gamma,
         std::size_t pairs, std::size_t corCells, std::size_t corDim,
         int * loc)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor");

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching lookup of maximumc correlation"
        << pyre::journal::endl;

    // launch the threads
    #pragma omp parallel for
    for (std::size_t pairId=0; pairId < pairs; pairId++) 
       _maxcor(gamma, pairId, corCells, corDim, loc);


    // all done
    return;
}


// the SAT generation kernel
template <typename value_t>
void
_maxcor(const value_t * gamma,
      std::size_t pairId,    // the pair index to process
      std::size_t corCells,  // the number of cells in the correlation hyper-matrix
      std::size_t corDim,    // and its number of rows
      int * loc)
{
    // locate the beginning of my correlation matrix
    auto cor = gamma + pairId*corCells;
    // locate the beginning of my stats table
    auto myloc = loc + 2*pairId;

    // find maximum value location
    auto high = cor[0];
    std::size_t highCell = 0;
    // go through all the cells in my matrix
    for (std::size_t cell = 1; cell < corCells; cell++) {
        // get the current value
        auto value = cor[cell];
        // if it is higher than the current max
        if (value > high) {
            // save new highest cor and cell
            highCell = cell;
            high = value;
        }
    }


    // save
    myloc[0] = highCell / corDim;  // row
    myloc[1] = highCell % corDim;  // col

    // all done
    return;
}


// end of file
