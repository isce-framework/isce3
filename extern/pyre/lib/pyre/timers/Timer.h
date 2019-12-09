// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#if !defined(pyre_timers_Timer_h)
#define pyre_timers_Timer_h

namespace pyre {
    namespace timers {
        class Timer;
    }
}

// imported types
#include <string>

// get platform specific clock type
#if defined(mm_platforms_darwin) || defined(__config_platform_darwin)
#include "mach/Clock.h"
#elif defined(mm_platforms_linux) || defined(__config_platform_linux)
#include <pyre/algebra/BCD.h>
#include "posix/Clock.h"
#else
#include <pyre/algebra/BCD.h>
#include "epoch/Clock.h"
#endif

// the timer
class pyre::timers::Timer {
    //typedefs
public:
    using clock_t = Clock;
    using timer_t = Timer;
    using name_t = std::string;

    // interface
public:
    inline auto name() const -> name_t;

    inline auto start() -> Timer &;
    inline auto stop() -> Timer &;
    inline auto reset() -> Timer &;

    inline auto lap();  // read the elapsed time
    inline auto read(); // return the accumulated time

    // meta methods
public:
    inline Timer(name_t name);
    virtual ~Timer();

    // data members
private:
    name_t _name;
    clock_t::tick_t _start;
    clock_t::tick_t _accumulated;

    static clock_t _theClock;

    // disable these
private:
    Timer(const Timer &) = delete;
    const timer_t & operator= (const timer_t &) = delete;
};

// get the inline definitions
#define pyre_timers_Timer_icc
#include "Timer.icc"
#undef pyre_timers_Timer_icc

#endif

// end of file
