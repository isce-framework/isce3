// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// given a file named "grid.dat" in the current directory, use the high level interface to map
// it into memory

// portability
#include <portinfo>
// externals
#include <unistd.h>
// support
#include <pyre/memory.h>

// entry point
int main() {
    // the cell type
    typedef double cell_t;
    // desired size
    size_t page = ::getpagesize();

    // allocate a buffer
    cell_t * buffer = new cell_t[page];

    // if this succeeded
    if (buffer) {
        // turn on the info channel
        // pyre::journal::debug_t("pyre.memory.direct").activate();
        // create a view over the buffer
        pyre::memory::view_t<cell_t> v1 {buffer};
        // make a copy
        pyre::memory::view_t<cell_t> v2 {v1};
        // check that they point to the same memory location
        if (v2.data() != v1.data()) {
            // make a channel
            pyre::journal::firewall_t firewall("pyre.memory.view");
            // complain
            firewall
                << pyre::journal::at(__HERE__)
                << "view not properly copied:" << pyre::journal::newline
                << "  expected " << v1.data() << ", got " << v2.data()
                << pyre::journal::endl;
            // and bail
            return 1;
        }
    }

    // if all goes well, the following deallocation will not raise any exceptions...
    delete [] buffer;

    // all done
    return 0;
}

// end of file
