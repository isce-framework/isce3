#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that info channels by the same name share a common state
"""


def test():
    # access the package
    import journal
    # build a info channel
    info = journal.info("journal.test1")
    # verify that it is on by default
    assert info.active == True
    # disable it
    info.active = False

    # access the same channel through another object
    clone = journal.info("journal.test1")
    # verify that it is now off
    assert clone.active == False

    # build a info channel with a different name
    another = journal.info("journal.test2")
    # verify that it is on by default, to make sure that there is no crosstalk between channels
    assert another.active == True

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
