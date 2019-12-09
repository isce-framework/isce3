#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the simple locator returns the correct location tag
"""


def script():
    import pyre.tracking

    locator = pyre.tracking.simple(source="simple")

    assert str(locator) == "simple"

    return locator


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    script()


# end of file
