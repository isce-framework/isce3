#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Access timers through the pyre executive
"""


def test():
    # access
    import pyre

    # make one
    t = pyre.executive.newTimer(name="test")

    # start it
    t.start()
    # stop it
    t.stop()
    # read it
    assert t.read() != 0

    # start it again
    t.start()
    # take a lap reading
    t.lap()
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
