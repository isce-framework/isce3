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
#include <complex>
#include <pyre/journal.h>
#include <pyre/memory.h>
#include <pyre/grid.h>

// main
int main() {
    // journal control
    pyre::journal::debug_t debug("pyre.memory.direct");
    // debug.activate();

    // easy access to the complex literal markup
    using namespace std::literals::complex_literals;

    // space
    using dcell_t = double;
    using ccell_t = std::complex<double>;
    // layout
    using rep_t = std::array<int, 2>;
    using index_t = pyre::grid::index_t<rep_t>;
    using layout_t = pyre::grid::layout_t<index_t>;
    // storage
    using cmem_t = pyre::memory::heap_t<ccell_t>;
    using dmem_t = pyre::memory::heap_t<dcell_t>;
    // grid
    using dgrid_t = pyre::grid::grid_t<dcell_t, layout_t, dmem_t>;
    using cgrid_t = pyre::grid::grid_t<ccell_t, layout_t, cmem_t>;

    // make an ordering
    layout_t::packing_type packing {1u, 0u};
    // make a layout
    layout_t::shape_type shape {3, 3};
    // make a layout
    layout_t layout {shape, packing};

    // make a source grid
    cgrid_t cgrid {layout};
    // and a destination grid
    dgrid_t dgrid {layout};

    // make a view to the source grid
    cgrid_t::view_type cview = cgrid.view();
    // and a view to the destination grid
    dgrid_t::view_type dview = dgrid.view();

    // fill the source grid with data
    std::fill(cview.begin(), cview.end(), 1.0+1.0i);
    // fill the destination grid with the magnitudes of the source data
    std::transform(
                   cview.begin(), cview.end(),
                   dview.begin(),
                   [] (ccell_t c) -> dcell_t {return std::abs(c);}
                   );

    // make a channel
    pyre::journal::debug_t channel("pyre.grid");

    // loop over the grid
    for (auto idx : cgrid.layout()) {
        // get the value stored at this location in the source grid
        auto source = cgrid[idx];
        // get the value stored at this location in the destination grid
        auto destination = dgrid[idx];
        // if it's not what we expect
        if (destination != std::abs(source)) {
            // make a channel
            pyre::journal::error_t error("pyre.grid");
            // show me
            error
                << pyre::journal::at(__HERE__)
                << "abs(cgrid[" << idx << "]: " << source
                << ") != "
                << "dgrid[" << idx << "]: " << destination
                << pyre::journal::endl;
            // and bail
            return 1;
        }
        // show me
        channel
            << pyre::journal::at(__HERE__)
            << "cgrid[" << idx << "] = " << source << ", "
            << "dgrid[" << idx << "] = " << destination
            << pyre::journal::newline;
    }

    //  flush
    channel << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
