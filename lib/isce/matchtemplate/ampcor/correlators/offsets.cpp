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


// the kernel that assembles the offset field
template <typename value_t = float>
static void
_offsetField(const int * fine,         // the fine offsets
             std::size_t pairId,       // the tile id to process
             std::size_t margin,       // the origin of the coarse shifts
             std::size_t refineMargin, // origin of the refined shifts
             std::size_t zoom,         // the overall zoom factor of the refined shifts
             value_t * field             // results
             );


// run through the correlation matrix for each, find its maximum value and record its location
void
ampcor::kernels::
offsetField(const int * fine,         // the fine offsets
            std::size_t pairs,        // the total number of entries
            std::size_t margin,       // the origin of the coarse shifts
            std::size_t refineMargin, // origin of the refined shifts
            std::size_t zoom,         // the overall zoom factor of the refined shifts
            float * field             // results (which contains the coarse offsets)
            )
{
    // make a channel
    pyre::journal::debug_t channel("ampcor");

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching assembling of the offset fields tiles"
        << pyre::journal::endl;

    // launch
    #pragma omp parallel for
    for (std::size_t pairId=0; pairId < pairs; pairId++)
       _offsetField(fine, pairId, margin, refineMargin, zoom, field);
    
    // all done
    return;
}


// the kernel that assembles the offset field
template <typename value_t>
static void
_offsetField(const int * fine,          // the fine offsets
             std::size_t pairId,        // the tile id to process
             std::size_t margin,        // the origin of the coarse shifts
             std::size_t refineMargin,  // origin of the refined shifts
             std::size_t zoom,          // the overall zoom factor of the refined shifts
             value_t * field            // results
             )
{

    // a constant
    const value_t one = 1.0;
    // find the beginning of my {fine} offset
    auto myFine = fine + 2*pairId;
    // and the beginning of where i store my result
    auto myField = field + 2*pairId;

    // do the math
    myField[0] = (one*myField[0] - margin) + (one * myFine[0] / zoom - refineMargin);
    myField[1] = (one*myField[1] - margin) + (one * myFine[1] / zoom - refineMargin);

    // all done
    return;
}


// end of file
