#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the script locator returns the correct location tag
"""


def script():
    # get the package
    import pyre.tracking
    # make a locator
    locator = pyre.tracking.script(source=__file__, function="script", line=16)
    # check that the message is formatted correctly
    assert str(locator) == "file={!r}, line=16, function='script'".format(__file__)
    # all done
    return locator


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    script()


# end of file
