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
#include <exception>
#include <complex>
#include <string>
// pyre
#include <pyre/journal.h>
// local declarations
#include "kernels.h"

// helpers
template <typename value_t = float>
void
_refStats(value_t * rArena,
          std::size_t pairId,
          std::size_t refDim, std::size_t cellsPerTilePair,
          value_t * stats);


// compute the mean and std dev of reference tiles
void
ampcor::kernels::
refStats(float * rArena,
         std::size_t pairs, std::size_t refDim, std::size_t cellsPerTilePair,
         float * stats)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "arena has " << pairs << " blocks of " << cellsPerTilePair << " cells;"
        << " the reference tiles are " << refDim << "x" << refDim
        << pyre::journal::endl;

    #pragma omp parallel for
    for (std::size_t pairId = 0; pairId < pairs; pairId++)
       _refStats(rArena, pairId, refDim, cellsPerTilePair, stats);

    // all done
    return;
}


// implementations
template <typename value_t>
void
_refStats(value_t * rArena,
          std::size_t pairId,
          std::size_t refDim, std::size_t cellsPerTilePair,
          value_t * stats)
{

    // Find the start of current tile by skipping former tile pairs
    auto tile = rArena + pairId*cellsPerTilePair;

    // Compute the location of the cell past the end of my tile
    auto eot = tile + refDim*refDim;

    // Initialize the accumulator
    value_t sum = 0;

    // Compute sum over the current tile
    for (value_t * cell = tile; cell < eot; cell++) 
       sum += *cell;

    // Get the mean
    value_t mean = sum / (refDim*refDim);



    // Reiterate over the tile and compute the variance
    sum = 0;
    for (value_t * cell = tile; cell < eot; cell++) {
        // Get the cell value and subtract the mean
        auto value = *cell - mean;
        // Store it
        *cell = value;
        // Update the sum of the squares
        sum += value*value;
    }

    // Computes the *pseudo* standard deviation
    // (missing 1/N factor) 
    // and saves it in the output array
    stats[pairId] = std::sqrt(sum);

    // All done
    return;

}


// End of file
