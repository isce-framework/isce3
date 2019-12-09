#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that list conversions work as expected
"""


def test():
    import pyre.descriptors

    # create a descriptor
    descriptor = pyre.descriptors.list(schema=pyre.descriptors.str())

    # casts
    # successful
    assert [] == descriptor.coerce(())
    assert [] == descriptor.coerce([])
    assert [] == descriptor.coerce("()")
    assert [] == descriptor.coerce("[]")

    assert ["one"] == descriptor.coerce(("one",))
    assert ["one"] == descriptor.coerce(["one"])
    assert ["one"] == descriptor.coerce("[one]")
    assert ["one"] == descriptor.coerce("(one)")

    assert ["one", "two"] == descriptor.coerce(("one", "two"))
    assert ["one", "two"] == descriptor.coerce(["one", "two"])
    assert ["one", "two"] == descriptor.coerce("(one,two)")
    assert ["one", "two"] == descriptor.coerce("[one,two]")
    assert ["one", "two"] == descriptor.coerce("(one, two)")
    assert ["one", "two"] == descriptor.coerce("[one, two]")

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
