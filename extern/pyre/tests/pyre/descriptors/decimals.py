#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that decimal conversions work as expected
"""


def test():
    import decimal
    import pyre.descriptors

    # create a descriptor
    descriptor = pyre.descriptors.decimal()

    # check
    assert descriptor.coerce("1.20") == decimal.Decimal("1.20")

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
