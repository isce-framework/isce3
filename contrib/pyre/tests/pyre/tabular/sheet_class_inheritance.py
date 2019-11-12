#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Build a table hierarchy with single inheritance
"""


def test():
    import pyre.tabular

    class raw(pyre.tabular.sheet):
        """
        The sheet layout
        """

        sku = pyre.tabular.measure()
        production = pyre.tabular.measure()
        shipping = pyre.tabular.measure()
        margin = pyre.tabular.measure()
        overhead = pyre.tabular.measure()
        discount = pyre.tabular.measure()

    class pricing(raw):

        cost = raw.production + raw.shipping
        msrp = (1 + raw.margin + raw.overhead)*cost
        price = msrp*(1 - raw.discount)


    # check the base
    assert raw.pyre_name == "raw"
    assert identical(raw.pyre_localFields, (
        raw.sku, raw.production, raw.shipping, raw.margin, raw.overhead, raw.discount
        ))
    assert identical(raw.pyre_fields, raw.pyre_localFields)
    assert identical(raw.pyre_fields, (
        raw.sku, raw.production, raw.shipping, raw.margin, raw.overhead, raw.discount
        ))
    assert raw.pyre_derivations == ()


    # check the subclass
    assert pricing.pyre_name == "pricing"

    assert identical(pricing.pyre_localFields, (
        pricing.cost, pricing.msrp, pricing.price
        ))
    assert identical(pricing.pyre_fields, (
        pricing.sku, pricing.production, pricing.shipping, pricing.margin,
        pricing.overhead, pricing.discount,
        pricing.cost, pricing.msrp, pricing.price
        ))
    assert identical(pricing.pyre_fields, (
        pricing.sku, pricing.production, pricing.shipping, pricing.margin,
        pricing.overhead, pricing.discount,
        ))
    assert identical(pricing.pyre_derivations, (
        pricing.cost, pricing.msrp, pricing.price
        ))

    return pricing, raw


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
