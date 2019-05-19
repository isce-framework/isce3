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

// convenient type aliases
// the raster in this example
using slc_t = ampcor::dom::slc_t;
// the filename type
using uri_t = slc_t::uri_type;
// the shape type
using shape_t = slc_t::shape_type;

// the correlator
using correlator_t = ampcor::correlators::correlator_t<ampcor::dom::slc_t>;

// driver
int main() {

    // the name of the reference data file
    uri_t refName { "../../data/20061231.slc" };
    // its shape
    shape_t refShape { 36864ul, 10344ul };
    // make a raster
    slc_t reference(refName, refShape);

    // the chip shape in the reference window
    slc_t::index_type chip {128ul, 128ul};

    // define a tile in the reference image
    slc_t::index_type refBegin {100ul, 100ul};
    slc_t::index_type refEnd = refBegin + chip;
    // make a slice
    auto refSlice = reference.layout().slice(refBegin, refEnd);

    // the name of the target data file
    uri_t tgtName { "../../data/20061231.slc" };
    // its shape
    shape_t tgtShape { 36864ul, 10344ul };
    // make a raster
    slc_t target(tgtName, tgtShape);

    // the spread of the chip that forms the target search window
    slc_t::index_type spread = {32ul, 32ul};
    // define a tile
    slc_t::index_type tgtBegin = refBegin - spread;
    slc_t::index_type tgtEnd  = refEnd + spread;
    // build the target search window
    auto tgtSlice = target.layout().slice(tgtBegin, tgtEnd);

    // make a channel
    pyre::journal::debug_t channel("ampcor.correlators");

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "reference raster: " << pyre::journal::newline
        << "   shape: " << reference.layout().shape() << pyre::journal::newline
        << "  pixels: " << reference.pixels() << pyre::journal::newline
        << "   bytes: " << reference.size() << pyre::journal::newline
        << "reference slice: " << pyre::journal::newline
        << "    from: " << refSlice.low() << pyre::journal::newline
        << "      to: " << refSlice.high() << pyre::journal::newline
        << "   shape: " << refSlice.shape() << pyre::journal::newline
        << "target raster: " << pyre::journal::newline
        << "   shape: " << target.layout().shape() << pyre::journal::newline
        << "  pixels: " << target.pixels() << pyre::journal::newline
        << "   bytes: " << target.size() << pyre::journal::newline
        << "target slice: " << pyre::journal::newline
        << "    from: " << tgtSlice.low() << pyre::journal::newline
        << "      to: " << tgtSlice.high() << pyre::journal::newline
        << "   shape: " << tgtSlice.shape() << pyre::journal::newline
        << pyre::journal::endl;

    // make a view to the reference raster
    auto refView = reference.view(refSlice);
    // and a view to the target raster
    auto tgtView = target.view(tgtSlice);

    // make a correlator
    correlator_t ampcor { refView, tgtView };
    // form the correlation grid
    const auto & corr = ampcor.correlate();

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "correlation:" << pyre::journal::newline
        << "    shape: " << corr.layout().shape()
        << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
