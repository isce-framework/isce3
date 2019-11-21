#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that int conversions work as expected
"""


def test():
    import pyre.schemata

    # create a descriptor
    descriptor = pyre.schemata.int()

    # casts
    # successful
    assert 1 == descriptor.coerce(1)
    assert 1 == descriptor.coerce("1")
    # failures
    try:
        descriptor.coerce(test)
        assert False
    except descriptor.CastingError as error:
        pass

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
