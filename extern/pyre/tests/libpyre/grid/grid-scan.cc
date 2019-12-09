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
#include <numeric>
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
    typedef pyre::memory::heap_t<cell_t> heap_t;
    // grid
    typedef pyre::grid::grid_t<cell_t, layout_t, heap_t> grid_t;

    // make an ordering
    layout_t::packing_type packing {1u, 0u};
    // make a layout
    layout_t::shape_type shape {3, 3};
    // make a layout
    layout_t layout {shape, packing};

    // make the reference grid
    grid_t grid {layout};
    // make a view over the grid
    grid_t::view_type gview = grid.view();
    // make a value
    const double value = 2;
    // fill the grid with data
    std::fill(gview.begin(), gview.end(), value);

    // make the target grid
    grid_t sat {layout};
    // make a view over the grid
    grid_t::view_type sview = sat.view();
    // build the sum area table
    std::partial_sum(gview.begin(), gview.end(), sview.begin());

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");
    // sign in
    channel
        << pyre::journal::at(__HERE__)
        << pyre::journal::newline;

    // loop over the grid
    for (auto idx : sat.layout()) {
        // get the offset of this index
        auto offset = sat.layout().offset(idx);
        // get the value stored at this location
        auto stored = sat[idx];
        // the two should be related
        auto expected = value * (offset+1);
        // if not
        if (stored != expected) {
            // make a channel
            pyre::journal::error_t error("pyre.grid");
            // show me
            error
                << pyre::journal::at(__HERE__)
                << "grid[" << idx << "]: " << stored << " != " << expected
                << pyre::journal::endl;
            // and bail
            return 1;
        }

        // show it to me
        channel
            << "sat[" << idx << "] = " << stored << ", expected = " << expected
            << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
