// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise iterator loops
//   N.B.: this is a test of the implementation, not an example of the
//   recommended way to use iterators

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
    // create a shortcut to my target iterator type
    typedef pyre::grid::iterator_t<slice_t> iterator_t;

    // make a packing strategy
    slice_t::packing_type packing {1u, 0u};
    // build the iteration boundaries
    slice_t::index_type low {0, 0};
    slice_t::index_type high {3, 2};
    // make a slice
    slice_t slice {low, high, packing};
    // make a iterator
    iterator_t iterator {slice};

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");
    // sign in
    channel << pyre::journal::at(__HERE__);
    // loop until the iterator reaches the end
    // this test works because {cursor} is a reference to the {iterator} current value
    // this is not an example of the recommended way to use slice iterators...
    for (const auto & cursor = *iterator; cursor != slice.high(); ++iterator) {
        // show me
        channel << "  (" << cursor << ")" << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
