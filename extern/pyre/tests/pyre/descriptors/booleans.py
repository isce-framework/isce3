#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that boolean conversions work as expected
"""


def test():
    import pyre.descriptors

    # create a descriptor
    descriptor = pyre.descriptors.bool()

    # casts
    # successful
    assert True == descriptor.coerce(True)
    assert True == descriptor.coerce("y")
    assert True == descriptor.coerce("on")
    assert True == descriptor.coerce("ON")
    assert True == descriptor.coerce("yes")
    assert True == descriptor.coerce("YES")
    assert True == descriptor.coerce("true")
    assert True == descriptor.coerce("TRUE")

    assert False == descriptor.coerce(False)
    assert False == descriptor.coerce("n")
    assert False == descriptor.coerce("off")
    assert False == descriptor.coerce("OFF")
    assert False == descriptor.coerce("no")
    assert False == descriptor.coerce("NO")
    assert False == descriptor.coerce("false")
    assert False == descriptor.coerce("FALSE")

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
