// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// for the build system
#include <portinfo>

// packages
#include <cassert>
#include <map>
#include <vector>
#include <string>

// access to the low level inventory header file
#include <pyre/journal/Inventory.h>

// convenience
typedef pyre::journal::Inventory<true> true_t;
typedef pyre::journal::Inventory<false> false_t;

// main program
int main() {

    // instantiate a couple of inventories
    true_t on;
    false_t off;

    // check their default settings
    assert(on.state() == true);
    assert(off.state() == false);

    // flip them
    on.deactivate();
    off.activate();

    // check again
    assert(on.state() == false);
    assert(off.state() == true);

    // all done
    return 0;
}

// end of file
