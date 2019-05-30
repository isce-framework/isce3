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
    grid_t::shape_type shape {5ul, 6ul};
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

    // movement
    sumarea_t::index_type h {1ul, 0ul}; // horizontal movement
    sumarea_t::index_type v {0ul, 1ul}; // vertical movement
    sumarea_t::index_type d {1ul, 1ul}; // diagonal movement

    // pick a region; start with the whole grid
    auto top = client.layout().low();
    auto bot = client.layout().high();
    // make a slice out of the whole grid
    auto whole = client.layout().slice(top, bot);
    // compute the sum
    auto sum = sat.sum(whole);
    // it should be equal to the value of the bottom right corner of the sum table
    auto expected = sat[bot - d];
    // check
    if (sum != expected) {
        // make a channel
        pyre::journal::error_t error("pyre.grid");
        // show me
        error
            << pyre::journal::at(__HERE__)
            << "sum(sat) = " << sum << " != " << expected
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // exclude the top left corner
    auto interior = client.layout().slice(top+d, bot);
    // get the extent
    auto rect = bot - top - d;
    // compute the size
    size_t area = 1;
    for (auto sz : rect) {
        area *= sz;
    }

    // compute the sum
    sum = sat.sum(interior);
    // it should be equal to area of the slice
    expected = area;
    // check
    if (sum != expected) {
        // make a channel
        pyre::journal::error_t error("pyre.grid");
        // show me
        error
            << pyre::journal::at(__HERE__)
            << "sum(sat) = " << sum << " != " << expected
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // exclude the left column
    auto leftcolumn = client.layout().slice(top+h, bot);
    // get the extent
    auto lrect = bot - top - h;
    // compute the size
    size_t larea = 1;
    for (auto sz : lrect) {
        larea *= sz;
    }

    // compute the sum
    sum = sat.sum(leftcolumn);
    // it should be equal to area of the slice
    expected = larea;
    // check
    if (sum != expected) {
        // make a channel
        pyre::journal::error_t error("pyre.grid");
        // show me
        error
            << pyre::journal::at(__HERE__)
            << "sum(sat) = " << sum << " != " << expected
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // exclude the top row
    auto toprow = client.layout().slice(top+v, bot);
    // get the extent
    auto trect = bot - top - v;
    // compute the size
    size_t tarea = 1;
    for (auto sz : trect) {
        tarea *= sz;
    }

    // compute the sum
    sum = sat.sum(toprow);
    // it should be equal to area of the slice
    expected = tarea;
    // check
    if (sum != expected) {
        // make a channel
        pyre::journal::error_t error("pyre.grid");
        // show me
        error
            << pyre::journal::at(__HERE__)
            << "sum(sat) = " << sum << " != " << expected
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // all done
    return 0;
}


// end of file
