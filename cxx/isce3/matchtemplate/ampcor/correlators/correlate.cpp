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
// pull the declarations
#include "kernels.h"

// the correlation kernel
template <typename value_t = float>
void
_correlate(const value_t * arena,
           std::size_t pairId,
           const value_t * refStats,
           const value_t * tgtStats,
           std::size_t rdim, std::size_t rcells,
           std::size_t tdim, std::size_t tcells,
           std::size_t cdim, std::size_t ccells,
           value_t * correlation);


// implementation
void
ampcor::kernels::
correlate(const float * dArena, const float * refStats, const float * tgtStats,
          std::size_t pairs,
          std::size_t refCells, std::size_t tgtCells, std::size_t corCells,
          std::size_t refDim, std::size_t tgtDim, std::size_t corDim,
          float * dCorrelation)
{

    // make a channel
    pyre::journal::debug_t channel("ampcor");

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching multithreading on " << pairs << " pairs of tiles, with one thread "
        << "per pair "
        << pyre::journal::endl;

    #pragma omp parallel for
    for (std::size_t pairId=0; pairId < pairs; pairId++)
        _correlate(dArena, pairId, refStats, tgtStats, 
                   refDim, refCells, tgtDim, tgtCells, corDim, corCells,
                   dCorrelation);


    // all done
    return;
}


// the correlation kernel
template <typename value_t>
void
_correlate(const value_t * arena, // the dataspace
           std::size_t pairId, // the tile id
           const value_t * refStats, // std dev (unormalized) of the ref tile
           const value_t * tgtStats, // the mean table of the target tile
           std::size_t rdim, std::size_t rcells, // ref grid shape and size
           std::size_t tdim, std::size_t tcells, // tgt grid shape and size
           std::size_t cdim, std::size_t ccells, // cor grid shape and size
           value_t * correlation)
{


    // reference and target grids are interleaved; compute the stride
    std::size_t stride = rcells + tcells;

    // my {ref} starting point
    auto ref = arena + pairId*stride;

    // my {tgt} starting point 
    auto tgt = arena + pairId*stride + rcells;


    // Standard Pearson correlation coefficient implementation:

    // the numerator term
    value_t numerator = 0;
    // the target variance accumulator
    value_t tgtVariance = 0;

    // go through all possible row offsets for the sliding window
    for (std::size_t row = 0; row < cdim; row++) {
        // and all possible column offsets
        for (std::size_t col = 0; col < cdim; col++) {

           // look up the mean target amplitude
           auto mean = tgtStats[pairId*ccells + row*cdim + col];

           // initialize numerator and tgt variance for the 
           // current position {row, col} of the sliding window
           numerator = 0;
           tgtVariance = 0;

           // offset in tgt at current {row, col} pos
           auto offset = row*tdim + col;

           // go through all cell of ref window
           for (std::size_t idy=0; idy<rdim; idy++) {
               for (std::size_t idx=0; idx<rdim; idx++) {
                   // get current ref cell 
                   value_t r = ref[idy*rdim + idx];
                   // get current tgt cell and mean normalize it
                   value_t t = tgt[offset + idy*tdim + idx] - mean;
                   // update the numerator
                   numerator += r * t;
                   // and the target variance
                   tgtVariance += t * t;
               }
           }    

           // looks up the sqrt of the reference tile variance
           value_t refVariance = refStats[pairId];
           // computes the correlation
           auto corr = numerator / (refVariance * std::sqrt(tgtVariance));
           // computes the slot where this result goes
           std::size_t slot = pairId*ccells + row*cdim + col;
           // and writes the sum to the result vector
           correlation[slot] = corr;
        }
    }    


    // all done
    return;
}


// end of file
