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

    // the data file is two pages long so it fits
    size_t size = 2*page/sizeof(cell_t);
    // map the file
    // turn on the info channel
    // pyre::journal::debug_t("pyre.memory.direct").activate();
    // map a buffer over the file; it gets unmapped on destruction
    pyre::memory::direct_t<cell_t> map {name, size};

    // ask the map for its size and compare against our calculation
    if (map.size() != size) {
        // make a channel
        pyre::journal::firewall_t firewall("pyre.memory.direct");
        // complain
        firewall
            << pyre::journal::at(__HERE__)
            << "cell count mismatch for file '" << name << "': " << pyre::journal::newline
            << "  expected " << size << " cells, got " << map.size() << " cells"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // ask the map for its memory footprint and compare against our expectations
    if (map.bytes() != 2*page) {
        // make a channel
        pyre::journal::firewall_t firewall("pyre.memory.direct");
        // complain
        firewall
            << pyre::journal::at(__HERE__)
            << "size mismatch for file '" << name << "': " << pyre::journal::newline
            << "  expected " << (2*page) << " bytes, got " << map.bytes() << " bytes"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // all done
    return 0;
}

// end of file
