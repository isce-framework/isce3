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


// the nudge generation kernel
template <typename value_t = float>
static void
_nudge(std::size_t pairId,    // the tile id
       std::size_t oldDim,    // the old shape of the target tiles
       std::size_t newDim,    // the new shape of the target tiles
       std::size_t margin,    // the new margin of the search window
       int * loc,
       value_t * offset);


// Depending on the coarse maximum location on the grid, shift the location of the refined target tile
// so that it fits within the target data.
void
ampcor::kernels::
nudge(std::size_t pairs,     // the total number of tiles
      std::size_t refDim,    // the shape of the reference tiles
      std::size_t tgtDim,    // the shape of the target tiles
      std::size_t margin,    // the new margin around the reference tile
      int * loc,             // input/output: the updated maxcor location for upcoming refinement step
      float * offset)        // output: the offset (from coarse maxcor location) adjusted for nudging 
{
    // make a channel
    pyre::journal::debug_t channel("ampcor");

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching nudging of maxima locations"
        << pyre::journal::endl;

    // launch the kernels
    #pragma omp parallel for
    for (std::size_t pairId=0; pairId<pairs; pairId++)
       _nudge(pairId, tgtDim, refDim+2*margin, margin, loc, offset);

    // all done
    return;
}


// the nudging kernel
template <typename value_t>
static void
_nudge(std::size_t pairId,    // the tile index
       std::size_t oldDim,    // the shape of the offset target tiles
       std::size_t newDim,    // the shape of the refined target tiles
       std::size_t margin,    // the new margin of the search window
       int * loc,
       value_t * offset)
{

    // locate the beginning of my stats table
    auto myOffset = offset + 2*pairId;
    auto myloc = loc + 2*pairId;
    // read my position
    int row = myloc[0];
    int col = myloc[1];


    // let's do column nudging first
    myOffset[1] = col ; 
    int left = col - static_cast<int>(margin);
    // if it sticks out in the left move it to the far left and correct the 
    // offset accordingly.
    if (left < 0) { 
        myOffset[1] -= left; 
        left = 0;
    }
    
    // if it sticks out on the right move so that it fits and correct the offset
    //offset accordingly.
    if (left + newDim > oldDim) {
        myOffset[1] -= (left + static_cast<value_t>(newDim) - static_cast<value_t>(oldDim));
        left = oldDim - newDim;
    }

    // repeat for row
    myOffset[0] = row;
    int top = row - static_cast<int>(margin);
    // if it sticks outside the top of the tile move it to the top row and 
    // correct the offset offset accordingly.
    if (top < 0) {
        myOffset[0] -= top;
        top = 0;
    }
    
    // if it sticks out below the bottom row move it up so it fits and correct
    // the offset offset accordingly.
    if (top + newDim > oldDim) {
        myOffset[0] -= (top + static_cast<value_t>(newDim) - static_cast<value_t>(oldDim));
        top = oldDim - newDim;
    }
    

    // write the new locations
    myloc[0] = top;
    myloc[1] = left;

    // all done
    return;
}


// end of file
