#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Instantiate a simple immutable record using keywords
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
        overhead = pyre.records.measure()
        price = pyre.records.measure()


    # build a record
    r = record.pyre_immutable(
        sku="9-4013", description="organic kiwi", cost=.85, overhead=.15, price=1.0)

    # check
    assert r.sku == "9-4013"
    assert r.description == "organic kiwi"
    assert r.cost == .85
    assert r.overhead == .15
    assert r.price == 1.0

    return r


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
