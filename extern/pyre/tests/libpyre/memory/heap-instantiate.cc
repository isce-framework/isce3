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
#include <pyre/journal.h>
#include <pyre/memory.h>

// entry point
int main() {
    // the cell type
    typedef double cell_t;
    // desired size
    size_t page = ::getpagesize();
    // make an allocation
    pyre::memory::heap_t<cell_t> heap(page);

    // if all goes well, no exceptions will be thrown when these objects get destroyed
    return 0;
}

// end of file
