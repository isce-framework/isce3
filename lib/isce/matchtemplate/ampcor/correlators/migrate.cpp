// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>
// pyre
#include <pyre/journal.h>
// pull the declarations
#include "kernels.h"
#include <cstring>


// the migration kernel
template <typename pixel_t = std::complex<float>>
void
_migrate(const pixel_t * coarse,
         std::size_t pairId,
         std::size_t cellsPerPair, std::size_t cellsPerRefinedPair,
         std::size_t refCells, 
         std::size_t refRefinedCells, 
         std::size_t tdim, std::size_t edim, std::size_t trdim,
         const int * locations,
         pixel_t * refined);


// implementation. Moving reference and target tiles to new (enlarged)
// container for upcoming upsampling.
void
ampcor::kernels::
migrate(const std::complex<float> * coarse,
        std::size_t pairs,
        std::size_t refDim, std::size_t tgtDim, std::size_t expDim,
        std::size_t refRefinedDim, std::size_t tgtRefinedDim,
        const int * locations,
        std::complex<float> * refined)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor");

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching migration of the expanded maxcor "
        << "tiles to the refinement arena"
        << pyre::journal::endl;

    // shape calculations
    auto refCells = refDim * refDim;
    auto tgtCells = tgtDim * tgtDim;
    auto refRefinedCells = refRefinedDim * refRefinedDim;
    auto tgtRefinedCells = tgtRefinedDim * tgtRefinedDim;

    // so i can skip over work others are doing
    auto cellsPerPair = refCells + tgtCells;
    auto cellsPerRefinedPair = refRefinedCells + tgtRefinedCells;

    // launch
    #pragma omp parallel for
    for (std::size_t pairId=0; pairId<pairs; pairId++)
        _migrate(reinterpret_cast<const std::complex<float> *>(coarse),
                 pairId,
                 cellsPerPair, cellsPerRefinedPair,
                 refCells, refRefinedCells, 
                 tgtDim, expDim, tgtRefinedDim,
                 locations,
                 reinterpret_cast<std::complex<float> *>(refined));

    // all done
    return;
}


// the correlation kernel
template <typename pixel_t>
void
_migrate(const pixel_t * coarse,
         std::size_t pairId,
         std::size_t cellsPerPair, std::size_t cellsPerRefinedPair,
         std::size_t refCells, 
         std::size_t refRefinedCells, 
         std::size_t tdim, std::size_t edim, std::size_t trdim,
         const int * locations,
         pixel_t * refined)
{

    // unpack the location of the ULHC of my maxcor tile
    auto row = locations[2*pairId];
    auto col = locations[2*pairId + 1];

    // the source: (row, col) of the target tile of pair {b} in the coarse arena
    const pixel_t * src; 
    // the destination: the target tile of pair {b} in the refined arena
    pixel_t * dest;

    // go down the rows in tandem
    auto rowFootprint = edim * sizeof(pixel_t);
    for (std::size_t jdx = 0; jdx < edim; ++jdx) {
        // update the pointers to new row
        // source moves by a whole row in the target tile
        src  = coarse  + pairId*cellsPerPair + refCells + (row+jdx)*tdim + col;
        // destination moves by a whole row in the refined target tile
        dest = refined + pairId*cellsPerRefinedPair + refRefinedCells + jdx*trdim;
        // migrate data
        std::memcpy(dest, src, rowFootprint);
    }

    // all done
    return;
}


// end of file
