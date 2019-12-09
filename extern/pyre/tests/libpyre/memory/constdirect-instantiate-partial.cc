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
    // the name of the file
    pyre::memory::uri_t name {"grid.dat"};

    // map the file
    // we only want the second page of data, so it fits
    size_t size = page/sizeof(cell_t);
    // turn on the info channel
    // pyre::journal::debug_t("pyre.memory.direct").activate();
    // map a buffer over the file; it gets unmapped on destruction
    pyre::memory::constdirect_t<cell_t> map {name, size, page};

    // ask the map for its size and compare against our calculation
    if (map.size() != size) {
        // make a channel
        pyre::journal::firewall_t firewall("pyre.memory.direct");
        // complain
        firewall
            << pyre::journal::at(__HERE__)
            << "size mismatch for file '" << name << "': " << pyre::journal::newline
            << "  expected " << size << " cells, got " << map.size() << " cells"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // ask the map for its memory footprint and compare against our expectations
    if (map.bytes() != page) {
        // make a channel
        pyre::journal::firewall_t firewall("pyre.memory.direct");
        // complain
        firewall
            << pyre::journal::at(__HERE__)
            << "size mismatch for file '" << name << "': " << pyre::journal::newline
            << "  expected " << page << " bytes, got " << map.bytes() << " bytes"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // all done
    return 0;
}

// end of file
