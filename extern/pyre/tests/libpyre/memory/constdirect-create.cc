// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// access the low level interface to create a file that can fit a grid of a specified size
//
// N.B.: this test leaves behind a file named "grid.dat" that is used by the other tests; it
// must be cleaned up after the tests are run

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
    // the name of the file
    pyre::memory::uri_t name {"grid.dat"};

    // turn on the info channel
    // pyre::journal::debug_t("pyre.memory.direct").activate();
    // create a file that can fit the payload
    pyre::memory::constdirect_t<cell_t>::create(name, 2*page);

    // all done
    return 0;
}

// end of file
