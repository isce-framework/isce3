// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise grid index construction:
//   verify that all the parts are accessible through the public headers
//   verify constructor signatures
//   assemble an index
//   verify it can be injected into a stream

// portability
#include <portinfo>
// support
#include <pyre/grid.h>

// entry point
int main() {
    // fix the representation
    typedef std::array<int, 4> rep_t;
    // alias
    typedef pyre::grid::index_t<rep_t> index_t;
    // make an index
    index_t idx { 0, 1, 2, 3 };

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");
    // and display information about the tile index
    channel
        << pyre::journal::at(__HERE__)
        << "index : (" << idx << ")"
        << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
