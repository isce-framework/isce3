#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Instantiate a complex mutable record using the keyword form
"""


def test():
    import pyre.records

    class record(pyre.records.record):
        """
        A sample record
        """
        sku = pyre.records.measure()
        cost = pyre.records.measure()
        price = 1.25 * cost + .25


    # build a record
    r = record.pyre_mutable(sku="9-4013", cost=1.0)
    # check
    assert r.sku == "9-4013"
    assert r.cost == 1.0
    assert r.price == 1.25 * r.cost + .25

    # make a change
    r.cost = 2.0
    # verify it was saved
    assert r.cost == 2.0
    # verify that the price is evaluated correctly
    assert r.price == 1.25 * r.cost + .25

    return r


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
