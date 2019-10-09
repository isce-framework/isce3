// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise layout construction:
//   verify that all the parts are accessible through the public headers
//   verify constructor signatures
//   exercise the simpel interface

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
    typedef pyre::grid::layout_t<index_t> layout_t;

    // make a packing strategy
    layout_t::packing_type packing {3u, 2u, 1u, 0u};
    // make a shape
    layout_t::index_type shape {2, 3, 4, 5};
    // make a layout
    layout_t layout {shape, packing};

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");

    // display information about the layout shape and packing
    channel
        << pyre::journal::at(__HERE__)
        << "shape: (" << layout.shape() << ")" << pyre::journal::newline
        << "packing: (" << layout.packing() << ")" << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
