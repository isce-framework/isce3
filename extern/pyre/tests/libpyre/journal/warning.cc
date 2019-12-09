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

    // instantiate a warning channel
    pyre::journal::warning_t warning("pyre.journal.test");
    // check that it is inactive, by default
    assert(warning.isActive() == true);

    // activate it
    warning.deactivate();
    // and check
    assert(warning.isActive() == false);

    // now, instantiate again using the same channel name
    pyre::journal::warning_t again("pyre.journal.test");
    // check that it is active
    assert(again.isActive() == false);

    // all done
    return 0;
}

// end of file
