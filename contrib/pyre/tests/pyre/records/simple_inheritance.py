#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify record building in the presence of inheritance
"""


def test():
    import pyre.records

    class item(pyre.records.record):
        """
        A sample record
        """
        sku = pyre.records.measure()
        description = pyre.records.measure()

    class pricing(item):
        cost = pyre.records.measure()
        overhead = pyre.records.measure()
        price = pyre.records.measure()


    # explore the base
    assert isinstance(item.sku, pyre.records.measure)
    assert isinstance(item.description, pyre.records.measure)

    assert identical(item.pyre_localFields, (item.sku, item.description))
    assert identical(item.pyre_fields, (item.sku, item.description))
    assert identical(item.pyre_measures, (item.sku, item.description))
    assert identical(item.pyre_derivations, ())

    assert item.pyre_index[item.sku] == 0
    assert item.pyre_index[item.description] == 1

    # explore the derived class
    assert isinstance(pricing.sku, pyre.records.measure)
    assert isinstance(pricing.description, pyre.records.measure)
    assert isinstance(pricing.cost, pyre.records.measure)
    assert isinstance(pricing.overhead, pyre.records.measure)
    assert isinstance(pricing.price, pyre.records.measure)

    assert identical(pricing.pyre_localFields, (pricing.cost, pricing.overhead, pricing.price))
    assert identical(
        pricing.pyre_fields,
        (
            pricing.sku, pricing.description,
            pricing.cost, pricing.overhead, pricing.price,
         ))
    assert identical(pricing.pyre_measures, pricing.pyre_fields)
    assert identical(pricing.pyre_derivations, ())

    assert pricing.pyre_index[pricing.sku] == 0
    assert pricing.pyre_index[pricing.description] == 1
    assert pricing.pyre_index[pricing.cost] == 2
    assert pricing.pyre_index[pricing.overhead] == 3
    assert pricing.pyre_index[pricing.price] == 4

    return item, pricing


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
