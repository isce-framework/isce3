#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Build a rudimentary table
"""


def test():
    # access the package
    import pyre.tabular
    # the sheet
    class pricing(pyre.tabular.sheet):
        """
        The sheet layout
        """
        # measures
        sku = pyre.tabular.measure()
        production = pyre.tabular.measure()
        shipping = pyre.tabular.measure()
        margin = pyre.tabular.measure()
        overhead = pyre.tabular.measure()
        discount = pyre.tabular.measure()
        # derivations
        cost = production + shipping
        msrp = (1 + margin + overhead)*cost
        price = msrp*(1 - discount)

    # check the name
    assert pricing.pyre_name == "pricing"

    # check the structure
    assert identical(pricing.pyre_localFields, (
        pricing.sku, pricing.production, pricing.shipping, pricing.margin,
        pricing.overhead, pricing.discount,
        pricing.cost, pricing.msrp, pricing.price,
        ))
    assert identical(pricing.pyre_fields, pricing.pyre_localFields)
    assert identical(pricing.pyre_fields, (
        pricing.sku, pricing.production, pricing.shipping, pricing.margin,
        pricing.overhead, pricing.discount,
        ))
    assert identical(pricing.pyre_derivations, (
        pricing.cost, pricing.msrp, pricing.price,
        ))

    # all done
    return pricing


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
