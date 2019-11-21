#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the default values get registered correctly
"""


def test():
    # get the descriptor package
    from pyre import descriptors
    # get the base metaclass
    from pyre.patterns.AttributeClassifier import AttributeClassifier

    # first, the harvesting metaclass
    class harvester(AttributeClassifier):

        def __new__(cls, name, bases, attributes):
            # harvest
            for entryName, entry in cls.pyre_harvest(attributes, descriptors.stem):
                # initialize
                entry.bind(name=entryName)
            # build the class record
            return super().__new__(cls, name, bases, attributes)

    # the client
    class client(metaclass=harvester):
        # some descriptors
        sku = descriptors.int(default=4503)
        cost = descriptors.decimal(default=2.34)
        weight = descriptors.dimensional(default='.5 * lb')


    # externals
    import decimal
    # get the units
    import pyre.units
    from pyre.units.mass import lb

    # get the defaults; must coerce them since this no longer happens by default
    sku = client.sku.coerce(client.sku.default)
    cost = client.cost.coerce(client.cost.default)
    weight = client.weight.coerce(client.weight.default)

    # check the default types
    assert type(sku) == int
    assert type(cost) == decimal.Decimal
    assert type(weight) == pyre.units.dimensional

    # check the default values
    assert sku == 4503
    assert cost == 2.34
    assert weight == .5*lb

    # all done
    return client


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
