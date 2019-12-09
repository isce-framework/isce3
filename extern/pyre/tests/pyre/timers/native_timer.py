#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Access the extension module implementation
"""


def test():
    # access
    from pyre.timers.NativeTimer import NativeTimer as timer

    # make one
    t = timer(name="test")

    # start it
    t.start()
    # stop it
    t.stop()
    # read it
    elapsed = t.read()
    assert type(elapsed) == float
    assert elapsed != 0

    # start it again
    t.start()
    # take a lap reading
    elapsed = t.lap()
    assert type(elapsed) == float
    assert elapsed != 0
    # stop it
    t.stop()

    # reset it
    t.reset()
    assert t.read() == 0

    return t


# main
if __name__ == "__main__":
    test()


# end of file
