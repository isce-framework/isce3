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

    // instantiate a debug channel that is supposed to be on due to the value of the DEBUG_OPT
    // environment variable
    pyre::journal::debug_t debug("pyre.journal.test");
    // verify that it is on
    assert(debug.isActive() == true);

    // deactivate it
    debug.deactivate();
    // and check
    assert(debug.isActive() == false);

    // now, instantiate again using the same channel name
    pyre::journal::debug_t again("pyre.journal.test");
    // check that it is inactive
    assert(again.isActive() == false);

    // all done
    return 0;
}

// end of file
