// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise looping through slices

// portability
#include <portinfo>
// support
#include <pyre/grid.h>

// entry point
int main() {
    // fix the rep
    typedef std::array<int, 2> rep_t;
    // aliases
    typedef pyre::grid::index_t<rep_t> index_t;
    typedef pyre::grid::slice_t<index_t> slice_t;

    // make a packing strategy
    slice_t::packing_type packing {1u, 0u};
    // build the iteration boundaries
    slice_t::index_type low {0, 0};
    slice_t::index_type high {3, 2};
    // make a slice
    slice_t slice {low, high, packing};

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");
    // sign in
    channel << pyre::journal::at(__HERE__);
    // loop through the slice
    for (auto cursor : slice) {
        // show me
        channel << "  (" << cursor << ")" << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
