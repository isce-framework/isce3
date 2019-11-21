#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that fraction conversions work as expected
"""


def test():
    import fractions
    import pyre.schemata

    # create a descriptor
    descriptor = pyre.schemata.fraction()

    # check
    assert descriptor.coerce("3/1") == fractions.Fraction("3/1")
    assert descriptor.coerce("1.20") == fractions.Fraction("1.20")

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
