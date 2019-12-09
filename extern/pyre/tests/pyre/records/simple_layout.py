#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Build a simple record and verify its structure
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


    # explore the record class
    assert isinstance(record.sku, pyre.records.measure)
    assert isinstance(record.description, pyre.records.measure)
    assert isinstance(record.cost, pyre.records.measure)
    assert isinstance(record.overhead, pyre.records.measure)
    assert isinstance(record.price, pyre.records.measure)

    assert record.pyre_localFields == (
        record.sku, record.description, record.cost, record.overhead, record.price)


    assert identical(record.pyre_fields, record.pyre_localFields)
    assert identical(record.pyre_measures, record.pyre_localFields)
    assert identical(record.pyre_derivations, ())

    assert record.pyre_index[record.sku] == 0
    assert record.pyre_index[record.description] == 1
    assert record.pyre_index[record.cost] == 2
    assert record.pyre_index[record.overhead] == 3
    assert record.pyre_index[record.price] == 4

    return record


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
