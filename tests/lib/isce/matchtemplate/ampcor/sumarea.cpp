// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// configuration
//#include <portinfo>
// support
#include <pyre/journal.h>
// access the correlator support
#include <isce/matchtemplate/ampcor/correlators.h>
// and the client raster types
#include <isce/matchtemplate/ampcor/dom.h>

// the sumarea pixel type
using pixel_t = float;
// a grid on the heap
using grid_t = ampcor::correlators::heapgrid_t<2, pixel_t>;
// alias the sum area table
using sumarea_t = ampcor::correlators::sumarea_t<grid_t>;

// driver
int main() {
    // make a shape
    grid_t::shape_type shape {5ul, 5ul};
    // make a grid
    grid_t client { shape };
    // and a view over it
    auto view = client.view();

    // pick a value
    pixel_t value = 1;
    // fill the grid with it
    std::fill(view.begin(), view.end(), value);

    // build the sum area  table
    sumarea_t sat(client);

    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators");
    // dump the amplitude grid and the sum area table
    for (auto idx : client.layout()) {
        // compute the expected value
        pixel_t expected = (idx[0] + 1)*(idx[1] + 1);
        // get the stored value
        pixel_t stored = sat[idx];

        // if there is a mismatch
        if (stored != expected) {
            // make a channel
            pyre::journal::error_t error("pyre.grid");
            // show me
            error
                << pyre::journal::at(__HERE__)
                << "sat[" << idx << "]: " << stored << " != " << expected
                << pyre::journal::endl;
            // and bail
            return 1;
        }

        channel
            << "client[" << idx << "] = " << client[idx]
            << " -> "
            << "sat[" << idx << "] = " << sat[idx]
            << pyre::journal::newline;
    }
    //  flush
    channel << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
