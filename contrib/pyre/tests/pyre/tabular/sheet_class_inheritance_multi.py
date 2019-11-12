#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Build a hierarchy of tables with multiple inheritance
"""


def test():
    import pyre.tabular

    class production(pyre.tabular.sheet):

        sku = pyre.tabular.measure()
        production = pyre.tabular.measure()

    class shipping(pyre.tabular.sheet):

        shipping = pyre.tabular.measure()

    class pricing(production, shipping):

        margin = pyre.tabular.measure()
        overhead = pyre.tabular.measure()
        discount = pyre.tabular.measure()

        cost = production.production + shipping.shipping
        msrp = (1 + margin + overhead)*cost
        price = msrp*(1 - discount)


    # check the bases
    assert production.pyre_name == "production"
    assert identical(production.pyre_localFields, ( production.sku, production.production ))
    assert identical(production.pyre_fields, ( production.sku, production.production ))
    assert identical(production.pyre_fields, ( production.sku, production.production ))
    assert production.pyre_derivations == ()

    assert shipping.pyre_name == "shipping"
    assert identical(shipping.pyre_localFields, ( shipping.shipping, ))
    assert identical(shipping.pyre_fields, ( shipping.shipping, ))
    assert identical(shipping.pyre_fields, ( shipping.shipping, ))
    assert shipping.pyre_derivations == ()

    # check the subclass
    assert pricing.pyre_name == "pricing"
    assert identical(pricing.pyre_localFields, (
        pricing.margin, pricing.overhead, pricing.discount,
        pricing.cost, pricing.msrp, pricing.price
        ))
    assert identical(pricing.pyre_fields, (
        pricing.shipping,
        pricing.sku, pricing.production,
        pricing.margin, pricing.overhead, pricing.discount,
        pricing.cost, pricing.msrp, pricing.price
        ))
    assert identical(pricing.pyre_fields, (
        pricing.shipping,
        pricing.sku, pricing.production,
        pricing.margin, pricing.overhead, pricing.discount,
        ))
    assert identical(pricing.pyre_derivations, (
        pricing.cost, pricing.msrp, pricing.price
        ))

    return pricing, shipping, production


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
