// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// for the build system
#include <portinfo>

// my header
#include "Display.h"

// the map
pyre::timers::Display::index_t pyre::timers::Display::_index;


// lookup a timer in the index and return it
pyre::timers::Display::timer_t &
pyre::timers::Display::
retrieveTimer(name_t name) {
    //
    // lookup {name} in the {_index} and return the associated timer
    // if not found, create a new one and index it under {name}
    timer_t * timer;
    // try to find the timer associated with this name in the index
    index_t::iterator lookup = _index.find(name);

    // if it's there
    if (lookup != _index.end()) {
        // grab it
        timer = lookup->second;
    // otherwise
    } else {
        // make a new one
        timer = new timer_t(name);
        // add it to the index
        _index[name] = timer;
    }
    // and return a reference to the timer
    return *timer;
}


// destructor
pyre::timers::Display::
~Display() {}

// end of file
