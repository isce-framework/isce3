// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise grid packing construction:
//   verify that all the parts are accessible through the public headers
//   verify constructor signatures
//   assemble a packing specification
//   verify it can be injected into a stream

// portability
#include <portinfo>
// support
#include <pyre/grid.h>

// entry point
int main() {
    // alias
    typedef pyre::grid::packing_t<4> packing_t;
    // make the interleaving
    packing_t packing = {0u, 1u, 2u, 3u};

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");
    // and display information about the tile packing
    channel
        << pyre::journal::at(__HERE__)
        << "packing : (" << packing << ")"
        << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
