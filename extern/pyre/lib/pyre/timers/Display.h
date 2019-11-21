// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#if !defined(pyre_timers_Display_h)
#define pyre_timers_Display_h


// The class Display is a wrapper around Timer that adds the ability to assign names to timers
// and register them with a local registry. Clients can retrieve timer instances by name, so
// that timer control can be accomplished throughout and application without having to pass
// timer instances around.

// place me in the proper namespace
namespace pyre {
    namespace timers {
        class Display;
    }
}

// imported types
#include <string>
#include "Timer.h"
#include <pyre/patterns/Registrar.h>

// declaration
class pyre::timers::Display {
    // types
public:
    using timer_t = Timer;
    using index_t = pyre::patterns::Registrar<timer_t>;
    using name_t = std::string;

    // interface
public:
    inline auto name() const;
    // start a timer
    inline auto start() -> Display & ;
    // stop a timer
    inline auto stop() -> Display & ;
    // zero out a timer
    inline auto reset() -> Display & ;

    // take a reading in seconds from a *running* timer
    inline auto lap();
    // get the number of seconds accumulated by a *stopped* timer
    inline auto read();

    // locate a timer given its name
    static timer_t & retrieveTimer(name_t name);

    // meta methods
public:
    virtual ~Display();
    inline Display(name_t name);

    // implementation details
    // data members
private:
    timer_t & _timer;
    static index_t _index;
};

// get the inline definitions
#define pyre_timers_Display_icc
#include "Display.icc"
#undef pyre_timers_Display_icc

#endif

// end of file
