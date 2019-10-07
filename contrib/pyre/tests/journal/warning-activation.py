#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that warning channels by the same name share a common state
"""


def test():
    # access the package
    import journal
    # build a warning channel
    warning = journal.warning("activation")
    # verify that it is off by default, deactivated from a configuration source
    assert warning.active == False
    # enable it
    warning.active = True

    # access the same channel through another object
    clone = journal.warning("activation")
    # verify that it is on
    assert clone.active == True
    # disable it
    clone.active = False

    # check that the other channel has been deactivated as well
    assert warning.active == False

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
