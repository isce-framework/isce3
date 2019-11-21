// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise grid index construction:
//   verify that all the parts are accessible through the public headers
//   verify constructor signatures
//   assemble indices and verify they can be compared

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
    // make some indices
    index_t one = {0, 1, 2, 3};
    index_t same = {0, 1, 2, 3};
    index_t different = {0, 1, 3, 3};

    // make a channel
    pyre::journal::error_t channel("pyre.grid");

    // check the ones that are supposed to be the same
    if (one != same) {
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "index mismatch"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // check the ones that are supposed to be different
    if (one == different) {
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "index match"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // all done
    return 0;
}

// end of file
