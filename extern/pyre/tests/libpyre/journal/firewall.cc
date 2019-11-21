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

    // instantiate a firewall channel
    pyre::journal::firewall_t firewall("pyre.journal.test");
    // check that it is inactive, by default
    assert(firewall.isActive() == true);

    // activate it
    firewall.deactivate();
    // and check
    assert(firewall.isActive() == false);

    // now, instantiate again using the same channel name
    pyre::journal::firewall_t again("pyre.journal.test");
    // check that it is active
    assert(again.isActive() == false);

    // all done
    return 0;
}

// end of file
