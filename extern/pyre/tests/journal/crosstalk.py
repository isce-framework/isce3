#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that channels by the same name but of different types are orthogonal
"""


def test():
    # access the package
    import journal

    # build a channel of each kind using the same name
    debug = journal.debug("journal.test")
    firewall = journal.firewall("journal.test")
    info = journal.info("journal.test")
    warning = journal.warning("journal.test")
    error = journal.error("journal.test")

    # check their expected default state
    assert debug.active == False
    assert firewall.active == True
    assert info.active == True
    assert warning.active == True
    assert error.active == True

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
