#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify the class record in the presence of multiple inheritance
"""


def test():
    import pyre.tabular

    # a few sheets
    class production(pyre.tabular.sheet):
        sku = pyre.tabular.measure()
        production = pyre.tabular.measure()


    class shipping(pyre.tabular.sheet):
        shipping = pyre.tabular.measure()


    class pricing(production, shipping):
        # measures
        margin = pyre.tabular.measure()
        overhead = pyre.tabular.measure()
        discount = pyre.tabular.measure()
        # derivations
        cost = production.production + shipping.shipping
        msrp = (1 + margin + overhead)*cost
        price = msrp*(1 - discount)


    # short names for the stuctural parts
    measure = pyre.tabular.measure
    derivation = pyre.tabular.derivation

    # verify ancestry
    assert issubclass(production, pyre.tabular.record)
    # verify the accessors
    assert isinstance(production.sku, measure)
    assert isinstance(production.production, measure)
    # and their indices
    assert production.pyre_index[production.sku] == 0
    assert production.pyre_index[production.production] == 1

    # verify ancestry
    assert issubclass(shipping, pyre.tabular.record)
    # verify the accessors
    assert isinstance(shipping.shipping, measure)
    # and their indices
    assert shipping.pyre_index[shipping.shipping] == 0

    # verify ancestry
    assert issubclass(pricing, pyre.tabular.record)
    # verify the accessors
    assert isinstance(pricing.margin, measure)
    assert isinstance(pricing.overhead, measure)
    assert isinstance(pricing.discount, measure)
    assert isinstance(pricing.shipping, measure)
    assert isinstance(pricing.sku, measure)
    assert isinstance(pricing.production, measure)
    assert isinstance(pricing.cost, derivation)
    assert isinstance(pricing.msrp, derivation)
    assert isinstance(pricing.price, derivation)
    # and their indices
    assert pricing.pyre_index[pricing.shipping] == 0
    assert pricing.pyre_index[pricing.sku] == 1
    assert pricing.pyre_index[pricing.production] == 2
    assert pricing.pyre_index[pricing.margin] == 3
    assert pricing.pyre_index[pricing.overhead] == 4
    assert pricing.pyre_index[pricing.discount] == 5
    assert pricing.pyre_index[pricing.cost] == 6
    assert pricing.pyre_index[pricing.msrp] == 7
    assert pricing.pyre_index[pricing.price] == 8

    return pricing, shipping, production


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
