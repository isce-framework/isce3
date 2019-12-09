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
    debug = journal.debug("journal.test1")
    # verify that it is off by default
    assert debug.active == False
    # enable it
    debug.active = True

    # access the same channel through another object
    clone = journal.debug("journal.test1")
    # verify that it is now on
    assert clone.active == True

    # build a debug channel with a different name
    another = journal.debug("journal.test2")
    # verify that it is off by default, to make sure that there is no crosstalk between channels
    assert another.active == False

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
