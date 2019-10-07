// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise grid packing construction:
//   verify that all the parts are accessible through the public headers
//   verify constructor signatures
//   assemble a packing strategy
//   verify it can be iterated

// portability
#include <portinfo>
// support
#include <pyre/grid.h>

// entry point
int main() {
    // alias
    typedef pyre::grid::packing_t<4> packing_t;
    // instantiate an packinging
    packing_t packing = {0u, 1u, 2u, 3u};

    // make a channel
    pyre::journal::error_t channel("pyre.grid");

    // check the values one by one
    for (packing_t::size_type i=0; i < packing.size(); ++i) {
        // check this one
        if (packing[i] != static_cast<packing_t::value_type>(i)) {
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "index mismatch: " << packing[i] << " != " << i
                << pyre::journal::endl;
            // and bail
            return 1;
        }
    }

    // all done
    return 0;
}

// end of file
