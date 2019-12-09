#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that descriptors can be harvested
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
            # remove them from the attributes
            for entry in pile: del attributes[entry.name]
            # build the class record
            record = super().__new__(cls, name, bases, attributes)
            # attach the pile
            record.pile = pile
            # all done
            return record

    # the client
    class client(metaclass=harvester):
        # some descriptors
        sku = descriptors.int(default=4503)
        cost = descriptors.float(default=2.34)
        weight = descriptors.dimensional(default='.5 * lb')
        price = 2*cost + .5

    # verify that the descriptors were harvested correctly
    assert [entry.name for entry in client.pile] == ['sku', 'cost', 'weight', 'price']

    # check that the descriptors have been removed
    try:
        client.sku
        assert False
    except AttributeError:
        pass

    try:
        client.price
        assert False
    except AttributeError:
        pass
    try:
        client.weight
        assert False
    except AttributeError:
        pass

    # all done
    return client


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
