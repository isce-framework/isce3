// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// for the build system
#include <portinfo>

// grab the timer objects
#include <pyre/timers.h>

// main program
int main() {
    // make a timer
    pyre::timer_t test("test");

    // exercise the interface
    // start the timer
    test.start();
    // stop the timer
    test.stop();
    // read the timer
    test.read();

    // all done
    return 0;
}

// end of file
