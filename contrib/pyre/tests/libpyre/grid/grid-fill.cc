// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>
// externals
#include <algorithm>
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
    typedef std::array<int, 2> rep_t;
    typedef pyre::grid::index_t<rep_t> index_t;
    typedef pyre::grid::layout_t<index_t> layout_t;
    // storage
    typedef pyre::memory::view_t<cell_t> view_t;
    // grid
    typedef pyre::grid::grid_t<cell_t, layout_t, view_t> grid_t;

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");

    // make an ordering
    layout_t::packing_type packing {1u, 0u};
    // make a layout
    layout_t::shape_type shape {3, 3};
    // make a layout
    layout_t layout {shape, packing};

    // allocate some memory
    cell_t * buffer = new cell_t[layout.size()];

    // make grid
    // N.B.: {buffer} is auto-converted into a view by the grid constructor
    grid_t grid {layout, buffer};

    // make a view over the grid
    grid_t::view_type view = grid.view();
    // make a value
    const double value = 1;
    // fill the grid with data
    std::fill(view.begin(), view.end(), value);

    // loop over the grid
    for (auto idx : grid.layout()) {
        // get the value stored at this location
        auto stored = grid[idx];
        // if it's not what we expect
        if (stored != value) {
            // make a channel
            pyre::journal::error_t error("pyre.grid");
            // show me
            error
                << pyre::journal::at(__HERE__)
                << "grid[" << idx << "]: " << stored << " != " << value
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
