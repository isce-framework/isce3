#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that derived descriptors get harvested correctly
"""


def test():
    # get the descriptor package
    from pyre import descriptors
    # get the base metaclass
    from pyre.patterns.AttributeClassifier import AttributeClassifier

    # first, the harvesting metaclass
    class harvester(AttributeClassifier):

        def __new__(cls, name, bases, attributes):
            # the pile
            pile = []
            # harvest
            for entryName, entry in cls.pyre_harvest(attributes, descriptors.stem):
                # initialize
                entry.bind(name=entryName)
                # and add them to the pile
                pile.append(entry)
            # build the class record
            record = super().__new__(cls, name, bases, attributes)
            # attach the pile
            record.pile = pile
            # and return it
            return record

    # a new descriptor
    class money(descriptors.decimal): pass

    # the client
    class client(metaclass=harvester):
        # some descriptors
        sku = descriptors.int(default=4503)
        cost = money(default=2.34)

    # verify that the descriptors were harvested correctly
    assert [entry.name for entry in client.pile] == ['sku', 'cost']

    # get the defaults
    sku = client.sku.coerce(client.sku.default)
    cost = client.cost.coerce(client.cost.default)

    # check the default value
    assert client.sku.default == 4503
    assert client.cost.default == 2.34

    # all done
    return client


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
