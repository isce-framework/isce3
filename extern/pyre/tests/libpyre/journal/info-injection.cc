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
#include <pyre/journal.h>

// main program
int main() {

    // instantiate a info channel
    pyre::journal::info_t info("pyre.journal.test");
    info.deactivate();

    // inject all the standard manipulators and built in types
    info
        << pyre::journal::at(__HERE__)
        << pyre::journal::set("key", "value")
        << "Hello world!" << pyre::journal::newline
        << 0 << pyre::journal::newline
        << 0.0 << pyre::journal::endl;

    info
        << pyre::journal::at(__HERE__)
        << (void *)&info << pyre::journal::newline
        << std::string("Hello world!")
        << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
