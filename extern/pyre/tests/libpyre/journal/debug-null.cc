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
#include <iostream>

#define WITHOUT_JOURNAL_DEBUG
#include <pyre/journal.h>

// main program
int main() {

    // instantiate a debug channel
    pyre::journal::debug_t debug("pyre.journal.test");
    debug.isActive();
    debug.activate();
    debug.deactivate();

    // inject all the standard manipulators and built in types
    debug
        << pyre::journal::at(__HERE__)
        << pyre::journal::set("key", "value")
        << "Hello world!" << pyre::journal::newline
        << 0 << pyre::journal::newline
        << 0.0 << pyre::journal::endl;

    debug
        << pyre::journal::at(__HERE__)
        << (void *)&debug << pyre::journal::newline
        << std::string("Hello world!")
        << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
