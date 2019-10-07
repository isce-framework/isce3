#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that float conversions work as expected
"""


def test():
    import pyre.descriptors

    # create a descriptor
    descriptor = pyre.descriptors.float()

    # casts
    # successful
    assert 1.2 == descriptor.coerce(1.2)
    assert 1.2 == descriptor.coerce("1.2")
    # failures
    try:
        descriptor.coerce('what?')
        assert False
    except descriptor.CastingError as error:
        assert str(error).startswith("could not coerce 'what?' into a float")

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
