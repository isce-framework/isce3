#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the package manager is accessible
"""


def test():
    import pyre
    # build the executive
    executive = pyre.executive

    # access the external package manager
    assert executive.host.packager is not None

    # all done
    return executive


# main
if __name__ == "__main__":
    # do...
    test()


# end of file
