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
    // layout
    typedef std::array<int, 3> rep_t;
    typedef pyre::grid::index_t<rep_t> index_t;
    typedef pyre::grid::layout_t<index_t> layout_t;
    // storage
    typedef pyre::memory::view_t<cell_t> view_t;
    // grid
    typedef pyre::grid::grid_t<cell_t, layout_t, view_t> grid_t;

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");

    // make an ordering
    layout_t::packing_type packing {2u, 1u, 0u};
    // make a layout
    layout_t::shape_type shape {6, 4, 2};
    // make a layout
    layout_t layout {shape, packing};

    // allocate some memory
    cell_t * buffer = new cell_t[layout.size()];
    // initialize the memory with predictable values
    for (layout_t::size_type i=0; i<layout.size(); ++i) {
        buffer[i] = i;
    }

    // make grid
    // N.B.: {buffer} is auto-converted into a view by the grid constructor
    grid_t grid {layout, buffer};

    // loop over the grid
    for (auto idx : grid.layout()) {
        // get the value stored at this location
        auto value = grid[idx];
        // the expected value is the current offset as a double
        grid_t::cell_type expected = grid.layout().offset(idx);
        // if they are not the same
        if (value != expected) {
            // make a channel
            pyre::journal::error_t error("pyre.grid");
            // show me
            error
                << pyre::journal::at(__HERE__)
                << "grid[" << idx << "]: " << value << " != " << expected
                << pyre::journal::endl;
            // and bail
            return 1;
        }
    }

    // clean up
    delete [] buffer;
    // all done
    return 0;
}


// end of file
