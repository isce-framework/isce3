#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the file locator returns the correct location tag
"""


def test():
    import pyre.tracking

    locator = pyre.tracking.file(source="script.py", line=16, column=2)

    assert str(locator) == "file='script.py', line=16, column=2"

    return locator


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
