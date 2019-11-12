// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// exercise layout construction:
//   verify that all the parts are accessible through the public headers
//   verify constructor signatures
//   instantiate a layout and verify it can be iterated
//   exercise the index <-> offset calculations

// portability
#include <portinfo>
// support
#include <pyre/grid.h>

// entry point
int main() {
    // fix the rep
    typedef std::array<int, 4> rep_t;
    // build the parts
    typedef pyre::grid::index_t<rep_t> index_t;
    typedef pyre::grid::layout_t<index_t> layout_t;

    // make a shape
    layout_t::index_type shape {2, 3, 4, 5};
    // make a layout with the default packing order
    layout_t layout {shape};

    // initialize the offset
    size_t offset = 0;

    // loop over the layout in packing order
    for (auto index : layout) {
        // get the offset of the pixel at this index
        auto pixel = layout[index];
        // verify it has the expected value
        if (offset != pixel) {
            // open a channel
            pyre::journal::error_t error("pyre.grid.index");
            // complain
            error
                << pyre::journal::at(__HERE__)
                << "offset error: " << offset << " != " << pixel
                << pyre::journal::endl;
            // and bail
            return 1;
        }

        // map the offset back to an index
        auto refl = layout[offset];
        // and verify it is identical to our loop index
        if (refl != index) {
            // open a channel
            pyre::journal::error_t error("pyre.grid.index");
            // complain
            error
                << pyre::journal::at(__HERE__)
                << "index error at offset " << offset << pyre::journal::newline
                << "(" << index << ") != (" << refl << ")"
                << pyre::journal::endl;
            // and bail
            return 1;
        }

        // update the counter
        offset++;
    }

    // all done
    return 0;
}

// end of file
