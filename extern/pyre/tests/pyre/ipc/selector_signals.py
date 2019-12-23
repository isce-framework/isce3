#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify the selector can handle signals
"""


def test():
    # externals
    import os
    import signal
    import itertools
    # support
    import journal
    import pyre.ipc
    from pyre.units.SI import second

    # make a channel
    channel = journal.debug("selector")
    # activate
    # channel.active = True
    # journal.debug("pyre.ipc.selector").active = True

    # get the process id
    pid = os.getpid()

    # instantiate a selector
    s = pyre.ipc.newSelector()

    # build a counter
    counter = itertools.count()

    # build a clock
    def cuckoo(timestamp):
        # grab the counter
        n = next(counter)
        # show me
        channel.log(f"{pid}: n={n}, time={timestamp}")
        # reschedule
        return 0.1*second

    # a stopper
    def kill(timestamp):
        # tell me
        channel.log(f"{pid}: sending SIGHUP")
        # send a signal
        os.kill(pid, signal.SIGHUP)
        # and bail
        return

    # and a handler for HUP
    def hup(signum, frame):
        # tell me
        channel.log(f"{pid}: SIGHUP received; shutting down")
        # tell the selector to shut down
        s.stop()
        # all done
        return

    # register the signal handler
    signal.signal(signal.SIGHUP, hup)
    # register the clock
    s.alarm(interval=1*second, call=cuckoo)
    # and the stopper
    s.alarm(interval=0.5*second, call=kill)
    # invoke the selector
    s.watch()

    # wait for it to finish and return it
    return s


# main
if __name__ == "__main__":
    test()


# end of file
