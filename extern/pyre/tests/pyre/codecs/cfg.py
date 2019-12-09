#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the package is accessible
"""


def test():
    import pyre.config

    # get the codec manager
    m = pyre.config.newConfigurator()
    # ask for a pml codec
    reader = m.codec(encoding="cfg")
    # and return the manager and the reader
    return m, reader


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
