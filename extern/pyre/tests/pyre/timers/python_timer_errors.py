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
    # access
    from pyre.timers.PythonTimer import PythonTimer as timer

    # make one
    t = timer(name="test")

    # read a timer that has not been started
    assert t.read() == 0
    # take a lap reading from a timer that has not been started
    try:
        t.lap()
        assert False
    except TypeError:
        pass

    return t


# main
if __name__ == "__main__":
    test()


# end of file
