// -*- C++ -*-
//
// michael a.g. aïvázis, bryan riel
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
    typedef std::array<int, 2> rep_t;
    typedef pyre::grid::index_t<rep_t> index_t;
    typedef pyre::grid::layout_t<index_t> layout_t;
    // storage
    typedef pyre::memory::view_t<cell_t> view_t;
    // grid
    typedef pyre::grid::grid_t<cell_t, layout_t, view_t> grid_t;

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");

    // make a common ordering
    layout_t::packing_type packing {1u, 0u};
    // make a shape for reference grid
    layout_t::shape_type ref_shape {6, 4};
    // make a layout
    layout_t ref_layout {ref_shape, packing};

    // allocate some memory for reference grid
    cell_t * buffer = new cell_t[ref_layout.size()];
    // initialize the memory with predictable values
    for (layout_t::size_type i = 0; i < ref_layout.size(); ++i) {
        buffer[i] = i;
    }
    // make reference grid
    grid_t ref_grid {ref_layout, buffer};

    // make a shape for a new grid
    layout_t::shape_type new_shape {2, 4};
    // make a layout
    layout_t new_layout {new_shape, packing};

    // allocate memory for new grid
    cell_t * new_buffer = new cell_t[new_layout.size()];
    // initialize the memory with zeros
    std::fill(new_buffer, new_buffer + new_layout.size(), 0.0);
    // make new grid
    grid_t new_grid {new_layout, new_buffer};

    // create slice indices for reference grid
    const index_t ref_low = {2, 0};
    const index_t ref_high = {4, 4};
    const layout_t::slice_type ref_slice = {ref_low, ref_high, packing};

    // create slice indices for new grid for setting values (slice whole grid for testing)
    const index_t low = {0, 0};
    const index_t high = {2, 4};
    const layout_t::slice_type slice = {low, high, packing};

    // assign values from reference view to new view
    new_grid.view(slice) = ref_grid.view(ref_slice);

    // loop over the grid
    const double bias = ref_low[0] * ref_shape[1];
    for (auto idx : new_grid.layout()) {
        // get the value stored at this location
        auto value = new_grid[idx];
        // the expect values is the current offset as a double plus a bias from view
        grid_t::cell_type expected = new_grid.layout().offset(idx) + bias;
        // if they are not the same
        if (value != expected) {
            // make a channel
            pyre::journal::error_t error("pyre.grid");
            // show me
            error
                << pyre::journal::at(__HERE__)
                << "new_grid[" << idx << "]: " << value << " != " << expected
                << pyre::journal::endl;
            // and bail
            return 1;
        }
    }

    // clean up
    delete [] buffer;
    delete [] new_buffer;
    // all done
    return 0;
}


// end of file
