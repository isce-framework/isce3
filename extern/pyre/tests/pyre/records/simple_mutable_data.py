#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Instantiate a simple mutable record using raw data
"""


def test():
    import pyre.records

    class record(pyre.records.record):
        """
        A sample record
        """
        sku = pyre.records.measure()
        description = pyre.records.measure()
        cost = pyre.records.measure()


    # build a record
    r = record.pyre_mutable(data=("9-4013", "organic kiwi", .85))
    # check
    assert r.sku == "9-4013"
    assert r.description == "organic kiwi"
    assert r.cost == .85

    # make a change to the cost
    r.cost = 1

    # and verify that it was stored
    assert r.cost == 1

    # all done
    return r


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
