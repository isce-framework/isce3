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
    // types for the mosaic cells, which are themselves grids
    // space
    typedef double cell_t;
    // shape
    typedef std::array<int, 3> crep_t;
    typedef pyre::grid::index_t<crep_t> cindex_t;
    typedef pyre::grid::layout_t<cindex_t> clayout_t;
    // storage
    typedef pyre::memory::heap_t<cell_t> heap_t;
    // grid
    typedef pyre::grid::grid_t<cell_t, clayout_t, heap_t> grid_t;

    // types for the mosaic itself
    // shape
    typedef std::array<int, 2> mrep_t; // the mosaic is a 2d arrangement
    typedef pyre::grid::index_t<mrep_t> mindex_t;
    typedef pyre::grid::layout_t<mindex_t> mlayout_t;
    // storage
    typedef pyre::memory::heap_t<grid_t> mheap_t;
    // mosaic
    typedef pyre::grid::grid_t<grid_t, mlayout_t, mheap_t> mosaic_t;

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");
    // activate it
    // channel.activate();

    // setup the shape of the mosaic
    mlayout_t mlayout { {2,2} };
    // setup the shape of the cell grids
    clayout_t clayout { {3,3,3}, {2u,1u,0u} };

    // instantiate the mosaic
    mosaic_t mosaic {mlayout};
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "mosaic:" << pyre::journal::newline
        << "  shape: " << mosaic.layout()
        << pyre::journal::endl;

    // initialize the mosaic cells by going through each of its cells
    for (auto idx : mosaic.layout()) {
        // getting the address of the current one
        grid_t * current = &mosaic[idx];
        // and placing a new grid there
        new (current) grid_t(clayout);
    }

    // show me
    channel << pyre::journal::at(__HERE__);
    // the contents of each cell
    for (auto idx : mosaic.layout()) {
        channel
            << "mosaic[" << idx << "]: "
            << mosaic[idx].layout() << " at " << &mosaic[idx]
            << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;

    // clean up
    for (auto idx : mosaic.layout()) {
        // invoke the destructor explicitly
        mosaic[idx].~grid_t();
    }

    // all done
    return 0;
}


// end of file
