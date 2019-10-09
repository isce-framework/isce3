#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that dimensional conversions work as expected
"""


def test():
    import pyre.schemata
    from pyre.units.SI import m, kg, s

    # create a descriptor
    descriptor = pyre.schemata.dimensional()

    # casts
    # successful
    assert m == descriptor.coerce("meter")
    assert 9.81*kg*m/s**2 == descriptor.coerce("9.81*kg*m/s**2")

    # failures
    try:
        descriptor.coerce(1)
        assert False
    except descriptor.CastingError as error:
        assert str(error) == "could not coerce 1 into a dimensional quantity"

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
