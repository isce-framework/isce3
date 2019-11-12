# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package provides the necessary infrastructure for timing applications.

There are no timer factories here. The pyre executive maintains access to a timer registrar,
and clients are expected to build timers through it. For example, the sequence

    import pyre
    t = pyre.executive.timer(name="test")
    t.start()
    .
    .
    .
    t.stop()
    elapsed = t.read()

produces a timer, and registers it under the name "test". Timers must be started before any
readings can take place. Stopping a timer prevents it from accumulating time, while {t.read}
returns the total number of seconds the timer has been active. You can only {read} timers that
have been stopped. If you want to peek at the accumulated time without interfering with the
time operation, use {lap}. Timers can be {reset} and reused as many times as you like.

Another interesting feature is that registered timers are available from anywhere in an
application. You can register a timer in one place, access it and start it in another, and stop
it and take a reading in a third, all without needing to pass around the variable. The timer
registry grants access to the same timer when it is asked for a timer of known name.
"""

# timer registry
def newTimerRegistrar():
    """
    Build a new timer registrar
    """
    from .Registrar import Registrar
    return Registrar()


# end of file
