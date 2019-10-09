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
//   verify it can be modified

// portability
#include <portinfo>
// support
#include <pyre/grid.h>

// entry point
int main() {
    // make a channel
    pyre::journal::error_t channel("pyre.grid");

    // fix the representation
    typedef std::array<int, 2> rep_t;
    // alias
    typedef pyre::grid::index_t<rep_t> index_t;

    // make an index
    index_t one {1, 1};
    // form the expected answer;
    index_t two {2, 2};

    // add
    index_t sum = one + one;
    // check
    if (sum != two) {
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "index addition error: " << "expected: " << two << ", got: " << sum
            << pyre::journal::endl;
    }

    // multiply
    index_t prod = 2 * one;
    if (prod != two) {
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "index scaling error: " << "expected: " << two << ", got: " << prod
            << pyre::journal::endl;
    }

    // all done
    return 0;
}

// end of file
