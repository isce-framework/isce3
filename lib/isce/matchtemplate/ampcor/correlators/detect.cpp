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


// compute the amplitude of the signal tiles, assuming pixels are of type std::complex<float>
void
ampcor::kernels::
detect(const std::complex<float> * cArena, std::size_t cells, float * rArena)
{

    // make a channel
    pyre::journal::debug_t channel("ampcor");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching multithreading to convert " << cells << " complex cells to "
        << cells << " real cells"
        << pyre::journal::endl;

    #pragma omp parallel for
    for (std::size_t pos = 0; pos < cells; pos++)
        rArena[pos] = std::abs(cArena[pos]);

    // all done
    return;
}


// end of file
