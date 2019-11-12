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
// access to the journal header file
#include <pyre/journal.h>

// main program
int main() {

    // instantiate a debug channel
    pyre::journal::debug_t debug("pyre.journal.test");
    // check that it is inactive, by default
    assert(debug.isActive() == false);

    // activate it
    debug.activate();
    // and check
    assert(debug.isActive() == true);

    // now, instantiate again using the same channel name
    pyre::journal::debug_t again("pyre.journal.test");
    // check that it is active
    assert(again.isActive() == true);

    // all done
    return 0;
}

// end of file
