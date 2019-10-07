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
    error = journal.error("journal.test1")
    # verify that it is on by default
    assert error.active == True
    # disable it
    error.active = False

    # access the same channel through another object
    clone = journal.error("journal.test1")
    # verify that it is now off
    assert clone.active == False

    # build a error channel with a different name
    another = journal.error("journal.test2")
    # verify that it is on by default, to make sure that there is no crosstalk between channels
    assert another.active == True

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
