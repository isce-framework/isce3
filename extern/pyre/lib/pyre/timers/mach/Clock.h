// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_timers_Clock_h)
#define pyre_timers_Clock_h

namespace pyre {
    namespace timers {
        class Clock;
    }
}

// imported symbols
#include <mach/mach_time.h>

// the clock
class pyre::timers::Clock {
    // typedefs
public:
    typedef uint64_t tick_t;
    typedef mach_timebase_info_data_t info_t;

    // interface
public:
    inline tick_t time() const;
    inline double elapsed(tick_t delta) const;

    // meta methods
public:
    inline Clock();
    inline ~Clock();

    // data members
private:
    info_t _info;

    // disable these
private:
    Clock(const Clock &);
    const Clock & operator= (const Clock &);
};

// get the inline definitions
#define pyre_timers_Clock_icc
#include "Clock.icc"
#undef pyre_timers_Clock_icc

#endif


// end of file
