// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise grid index construction:
//   verify that all the parts are accessible through the public headers
//   verify constructor signatures
//   assemble an index
//   verify it can be modified

// portability
#include <portinfo>
// support
#include <pyre/grid.h>

// entry point
int main() {
    // fix the representation
    typedef std::array<int, 4> rep_t;
    // alias
    typedef pyre::grid::index_t<rep_t> index_t;
    // make an index
    index_t index = {2, 2, 2, 2};

    // make a channel
    pyre::journal::error_t channel("pyre.grid");

    // adjust the values one by one
    for (index_t::size_type i=0; i < index.size(); ++i) {
        // to a sequence of consecutive integers
        index[i] = i;
    }

    // check the values one by one
    for (index_t::size_type i=0; i < index.size(); ++i) {
        // check this one
        if (index[i] != static_cast<index_t::value_type>(i)) {
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "index mismatch: " << index[i] << " != " << i
                << pyre::journal::endl;
            // and bail
            return 1;
        }
    }

    // all done
    return 0;
}

// end of file
