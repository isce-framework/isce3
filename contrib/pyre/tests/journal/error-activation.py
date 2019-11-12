#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that error channels by the same name share a common state
"""


def test():
    # access the package
    import journal
    # build a error channel
    error = journal.error("activation")
    # verify that it is off by default, deactivated from a configuration source
    assert error.active == False
    # enable it
    error.active = True

    # access the same channel through another object
    clone = journal.error("activation")
    # verify that it is on
    assert clone.active == True
    # disable it
    clone.active = False

    # check that the other channel has been deactivated as well
    assert error.active == False

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
