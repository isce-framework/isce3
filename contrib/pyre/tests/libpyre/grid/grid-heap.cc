// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>
// externals
#include <iostream>
#include <pyre/journal.h>
#include <pyre/memory.h>
#include <pyre/grid.h>

// main
int main() {
    // journal control
    pyre::journal::debug_t debug("pyre.memory.direct");
    // debug.activate();

    // space
    typedef double cell_t;
    // shape
    typedef std::array<int, 3> rep_t;
    typedef pyre::grid::index_t<rep_t> index_t;
    typedef pyre::grid::layout_t<index_t> layout_t;
    // storage
    typedef pyre::memory::heap_t<cell_t> heap_t;
    // grid
    typedef pyre::grid::grid_t<cell_t, layout_t, heap_t> grid_t;

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");

    // make an ordering
    layout_t::packing_type packing {2u, 1u, 0u};
    // make a shape
    layout_t::shape_type shape {6, 4, 2};
    // make a layout
    layout_t layout {shape, packing};

    // allocate some memory on the heap and make a grid
    grid_t grid {layout};

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << grid[{1,1,1}]
        << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
