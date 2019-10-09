#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise data conversions for complex immutable records
"""


def test():
    import pyre.records

    class record(pyre.records.record):
        """
        A sample record
        """
        sku = pyre.records.str()
        cost = pyre.records.float()
        price = 1.25 * cost + .25


    # build a record
    r = record.pyre_immutable(data=("9-4013", "1.0"))
    # check
    assert r.sku == "9-4013"
    assert r.cost == 1.0
    assert r.price == 1.25 * r.cost + .25

    return r


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
