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


// the target statistics generation kernel
template <typename value_t = float>
static void
_tgtStats(const value_t * sat,
          std::size_t pairId, std::size_t refDim, std::size_t tgtDim, std::size_t corDim,
          value_t * stats);


// implementation

// precompute the amplitude averages for all possible placements of the search tile within the
// target search window for all pairs in the plan. we allocate room for {_pairs}*{_corCells}
// floating point values and use the precomputed SAT tables.
//
// the SAT tables require a slice and produce the sum of the values of cells within the slice
// in no more than four memory accesses per search tile; there are boundary cases to consider
// that add a bit of complexity to the implementation; the boundary cases could have been
// trivialized using ghost cells around the search window boundary, but the memory cost is high
void
ampcor::kernels::
tgtStats(const float * dSAT,
         std::size_t pairs, std::size_t refDim, std::size_t tgtDim, std::size_t corDim,
         float * dStats)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor");

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching computation of target amplitude averages"
        << pyre::journal::endl;

    // launch the kernels
    #pragma omp parallel for
    for (std::size_t pairId=0; pairId < pairs; pairId++)
       _tgtStats(dSAT, pairId, refDim, tgtDim, corDim, dStats);

    // all done
    return;
}


// the SAT generation kernel
template <typename value_t>
void
_tgtStats(const value_t * dSAT,
      std::size_t pairId,    // the target tile index
      std::size_t refDim,    // the shape of each reference tile
      std::size_t tgtDim,    // the shape of each target tile
      std::size_t corDim,    // the shape of each grid
      value_t * dStats)
{

    // compute the number of cells in a reference tile
    auto refCells = refDim * refDim;
    // compute the number of cells in a target tile
    auto tgtCells = tgtDim * tgtDim;
    // compute the number of cells in each correlation matrix
    auto corCells = corDim * corDim;

    // locate the beginning of my SAT table
    auto sat = dSAT + pairId*tgtCells;
    // locate the beginning of my stats table
    auto stats = dStats + pairId*corCells;

    // go through all possible row offsets
    for (std::size_t row = 0; row < corDim; ++row) {
        // the row limit of the tile
        // this depends on the shape of the reference tile
        std::size_t rowMax = row + refDim - 1;

        // go through all possible column offsets
        for (std::size_t col = 0; col < corDim; ++col) {
            // the column limit of the tile
            //  this depends on the shape of the reference tile
            std::size_t colMax = col + refDim - 1;

            // initialize the sum by reading the bottom right corner; it's guaranteed to be
            // within the SAT
            value_t sum = sat[rowMax*tgtDim + colMax];

            // if the slice is not top-aligned
            // subtract the value from the upper right corner
            if (row > 0) 
                sum -= sat[(row-1)*tgtDim + colMax];

            // if the slice is not left-aligned
            // subtract the value of the upper left corner
            if (col > 0) 
                sum -= sat[rowMax*tgtDim + (col - 1)];

            // if the slice is not aligned with the upper left corner
            // restore its contribution to the sum
            if (row > 0 && col > 0) 
                sum += sat[(row-1)*tgtDim + (col-1)];
            
            // compute the offset that brings us to this placement in this tile
            std::size_t offset = row*corDim + col;

            // compute the average value and store it
            stats[offset] = sum / refCells;
        }
    }

    // all done
    return;
}


// end of file
