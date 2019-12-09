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
#include <string>
#include <cstdlib>

// access to the low level index header file
#include <pyre/journal/Index.h>

// convenience
typedef pyre::journal::Index<bool> index_t;

// main program
int main() {

    // instantiate an index
    index_t index;

    // request a key that is not there
    bool & state = index.lookup("test");
    // verify that this is off by default
    assert(state == false);
    // turn it on
    state = true;

    // ask for it again, this time read only
    bool again = index.lookup("test");
    // verify that it is now on
    assert(again == true);

    // all done
    return 0;
}

// end of file
