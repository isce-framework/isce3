// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise layout construction
//   verify that all the parts are accessible through the public headers
//   verify constructor signatures
//   verify layouts can be sliced

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

    // specify the slice region
    index_t begin {1,1,1,1};
    index_t end = layout.shape();

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");
    // if the channel is not active
    if (!channel) {
        // we are done
        return 0;
    }

    // otherwise, sign in
    channel << pyre::journal::at(__HERE__);
    // loop over the layout in packing order
    for (auto index : layout.slice(begin, end)) {
        // get the offset of the pixel at this index
        auto pixel = layout[index];
        // show me
        channel << "(" << index << ") -> " << pixel << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
