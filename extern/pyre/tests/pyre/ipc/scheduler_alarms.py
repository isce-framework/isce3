#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the scheduler can raise alarms
"""


def test():
    # access the package
    import pyre.ipc
    # instantiate a scheduler
    s = pyre.ipc.newScheduler()

    # get select
    import select
    # get time
    from time import time as now
    # get the units of time
    from pyre.units.SI import second
    # build a counter
    import itertools
    counter = itertools.count()
    # build a handler
    def handler(timestamp):
        n = next(counter)
        # print("n={}, time={}".format(n, timestamp))
        return

    # setup some alarms
    s.alarm(interval=0*second, call=handler)
    s.alarm(interval=1*second, call=handler)
    s.alarm(interval=0.5*second, call=handler)
    s.alarm(interval=0.25*second, call=handler)
    s.alarm(interval=0.75*second, call=handler)
    s.alarm(interval=0.3*second, call=handler)
    s.alarm(interval=0.5*second, call=handler)
    # how many?
    alarms = len(s._alarms)

    # forever
    while 1:
        # get the timeout
        timeout = s.poll()
        # if there are no more alarms scheduled
        if timeout is None:
            # bail out
            break
        # show me what's there
        # t = now()
        # print("timeout =", timeout, "alarms:", tuple(a.time-t for a in reversed(s._alarms)))
        # otherwise, go to sleep
        r,w,e = select.select([],[],[], timeout)
        # verify that {select} returned because the timeout expired
        assert r == [] and w == [] and e == []
        # raise any overdue alarms
        s.awaken()

    # verify that all alarms fired
    assert next(counter) == alarms

    # and return the scheduler
    return s


# main
if __name__ == "__main__":
    test()


# end of file
