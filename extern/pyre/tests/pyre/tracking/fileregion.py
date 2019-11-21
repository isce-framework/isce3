#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the file region locator returns the correct location tag
"""


def test():
    import pyre.tracking

    start = pyre.tracking.file(source="script.py", line=16, column=2)
    end = pyre.tracking.file(source="script.py", line=17, column=52)

    region = pyre.tracking.region(start=start, end=end)

    assert str(region) == "file='script.py', from (line=16, column=2) to (line=17, column=52)"

    return region


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
