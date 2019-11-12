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

    // instantiate a info channel
    pyre::journal::info_t info("pyre.journal.test");
    // check that it is active, by default
    assert(info.isActive() == true);

    // deactivate it
    info.deactivate();
    // and check
    assert(info.isActive() == false);

    // now, instantiate again using the same channel name
    pyre::journal::info_t again("pyre.journal.test");
    // check that it is inactive
    assert(again.isActive() == false);

    // all done
    return 0;
}

// end of file
