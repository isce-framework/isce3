#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Access the timer that is implemented on top of the time module form the python standard library
"""


def test():
    # externals
    import time
    # access
    from pyre.timers.PythonTimer import PythonTimer as timer

    # make one
    t = timer(name="test")

    # start it
    t.start()
    # sleep for a bit
    time.sleep(1)
    # stop it
    t.stop()
    # check
    assert t._accumulatedTime != 0
    # read it
    t.read()

    # start it again
    t.start()
    # take a lap reading
    t.lap()
    # stop it
    t.stop()

    # reset it
    t.reset()
    assert t._accumulatedTime == 0

    return t


# main
if __name__ == "__main__":
    test()


# end of file
