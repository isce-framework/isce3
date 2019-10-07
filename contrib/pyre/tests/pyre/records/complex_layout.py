#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Create a record that has a derived field
"""


def test():
    import pyre.records

    class item(pyre.records.record):
        """
        A sample record
        """
        cost = pyre.records.float()
        price = 1.25 * cost


    # explore the record class
    assert isinstance(item.cost, pyre.records.measure)
    assert isinstance(item.price, pyre.records.derivation)

    assert identical(item.pyre_fields, (item.cost, item.price))
    assert identical(item.pyre_measures, (item.cost,))
    assert identical(item.pyre_derivations, (item.price,))

    assert item.pyre_index[item.cost] == 0
    assert item.pyre_index[item.price] == 1

    # now instantiate one
    sample = item.pyre_mutable(cost=1.0)
    # check
    assert sample.cost == 1.0
    assert sample.price == 1.25

    return sample


def identical(s1, s2):
    """
    Verify that the nodes in {s1} and {s2} are identical. This has to be done carefully since
    we must avoid triggering __eq__
    """
    for n1, n2 in zip(s1, s2):
        if n1 is not n2: return False
    return True


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
