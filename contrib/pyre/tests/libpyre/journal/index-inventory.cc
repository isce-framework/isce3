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
// access to the packages used by Index
#include <map>
#include <vector>
#include <string>
#include <cstdlib>

// access to the low level header files
#include <pyre/journal/Inventory.h>
#include <pyre/journal/Index.h>

// convenience
typedef pyre::journal::Index< pyre::journal::Inventory<true> > critical_t;
typedef pyre::journal::Index< pyre::journal::Inventory<false> > boring_t;

// main program
int main() {

    // instantiate an index of inventory objects that are on by default
    critical_t critical;
    // request a key that is not there
    critical_t::value_t & on = critical.lookup("test");
    // verify that this is on by default
    assert(on.state() == true);
    // turn it off
    on.deactivate();
    // ask for it again, this time read only
    critical_t::value_t on_after = critical.lookup("test");
    // verify that it is now off
    assert(on_after.state() == false);

    // instantiate an index of inventory objects that are off by default
    boring_t boring;
    // request a key that is not there
    boring_t::value_t & off = boring.lookup("test");
    // verify that this is off by default
    assert(off.state() == false);
    // turn it on
    off.activate();
    // ask for it again, this time read only
    boring_t::value_t off_after = boring.lookup("test");
    // verify that it is now on
    assert(off_after.state() == true);

    // all done
    return 0;
}

// end of file
