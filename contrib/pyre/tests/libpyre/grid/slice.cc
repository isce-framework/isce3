// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise slice construction
//   verify that all the parts are accessible through the public headers
//   verify constructor signatures
//   instantiate and access the simple interface

// portability
#include <portinfo>
// support
#include <pyre/grid.h>

// entry point
int main() {
    // fix the rep
    typedef std::array<int, 4> rep_t;
    // build the parts
    typedef pyre::grid::index_t<rep_t> index_t;
    typedef pyre::grid::slice_t<index_t> slice_t;

    // make a packing strategy
    slice_t::packing_type packing {3u, 2u, 1u, 0u};
    // make a lower bound
    slice_t::index_type low {0, 0, 0, 0};
    // make an upper bound
    slice_t::index_type high {2, 3, 4, 5};
    // make a slice
    slice_t slice {low, high, packing};

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");

    // display information about the tile shape and packing
    channel
        << pyre::journal::at(__HERE__)
        << "low: (" << slice.low() << ")" << pyre::journal::newline
        << "high: (" << slice.high() << ")" << pyre::journal::newline
        << "packing: (" << slice.packing() << ")" << pyre::journal::newline
        << "shape: (" << slice.shape() << ")"
        << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
