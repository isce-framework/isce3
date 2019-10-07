#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that string conversions work as  expected
"""


def test():
    import pyre.descriptors

    # create a descriptor
    descriptor = pyre.descriptors.str()

    # casts
    assert "hello" == descriptor.coerce("hello")

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
