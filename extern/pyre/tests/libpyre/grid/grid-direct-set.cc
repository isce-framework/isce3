// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>
// externals
#include <numeric>
// support
#include <pyre/journal.h>
#include <pyre/memory.h>
#include <pyre/grid.h>

// main
int main() {
    // journal control
    // pyre::journal::debug_t debug("pyre.memory.direct");
    // debug.activate();

    // space
    typedef double cell_t;
    // shape
    typedef std::array<int, 3> rep_t;
    typedef pyre::grid::index_t<rep_t> index_t;
    typedef pyre::grid::layout_t<index_t> layout_t;
    // convenience
    typedef pyre::memory::uri_t uri_t;
    // grid
    typedef pyre::grid::directgrid_t<cell_t, layout_t> grid_t;

    // make an ordering
    layout_t::packing_type packing {2u, 1u, 0u};
    // make a shape
    layout_t::shape_type shape {6, 4, 5};
    // make a layout
    layout_t layout {shape, packing};

    // the name of the file
    uri_t name {"grid.dat"};
    // map it and make the grid
    grid_t grid {name, layout};

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");
    // loop over the grid
    for (auto idx : grid.layout()) {
        // reduce the index
        auto v = std::accumulate(idx.begin(), idx.end(), 1, std::multiplies<cell_t>());
        // and store the value
        grid[idx] = v;
        // show me
        channel
            << pyre::journal::at(__HERE__)
            << "grid[" << idx << "] = " << grid[idx]
            << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
