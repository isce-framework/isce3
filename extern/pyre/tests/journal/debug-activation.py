#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that debug channels by the same name share a common state
"""


def test():
    # access the package
    import journal
    # build a debug channel
    debug = journal.debug("activation")
    # verify that it is on by default, activated from a configuration source
    assert debug.active == True
    # disable it
    debug.active = False

    # access the same channel through another object
    clone = journal.debug("activation")
    # verify that it is off
    assert clone.active == False
    # enable it
    clone.active = True

    # check that the other channel has been activated as well
    assert debug.active == True

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
