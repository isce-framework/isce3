// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise grid packing construction:
//   verify that all the parts are accessible through the public headers
//   assemble a C style packing strategy
//   verify it can be injected into a stream

// portability
#include <portinfo>
// support
#include <pyre/grid.h>

// entry point
int main() {
    // alias
    typedef pyre::grid::packing_t<4> packing_t;
    // make a C-style interleaving
    packing_t packing = packing_t::rowMajor();

    // go through the packing contents
    for (packing_t::size_type i=0; i<packing.size(); ++i) {
        // we expect
        packing_t::value_type expected = packing.size() - i - 1;
        // and check that it is sorted in descending packing
        if (packing[i] != expected) {
            // make a channel
            pyre::journal::error_t channel("pyre.grid");
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "packing mismatch at " << i << ": " << packing[i] << " != " << expected
                << pyre::journal::endl;
            // and bail
            return 1;
        }
    }

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
