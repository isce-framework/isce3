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
    from pyre.descriptors import stem
    # get the base metaclass
    from pyre.patterns.AttributeClassifier import AttributeClassifier

    # first, the harvesting metaclass
    class harvester(AttributeClassifier):

        def __new__(cls, name, bases, attributes):
            # the pile
            pile = []
            # harvest
            for entryName, entry in cls.pyre_harvest(attributes, stem):
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

            # filter the measures
            record.measures = list(filter(lambda x: isinstance(x, stem.variable), pile))
            # and derivations
            record.derivations = list(filter(lambda x: isinstance(x, stem.operator), pile))
            # all done
            return record

    # get the package
    from pyre import descriptors
    # the client
    class client(metaclass=harvester):
        # some descriptors
        sku = descriptors.int(default=4503)
        cost = descriptors.float(default=2.34)
        weight = descriptors.dimensional(default='.5 * lb')
        price = 2*cost + .5

    # verify that the descriptors were harvested correctly
    assert [entry.name for entry in client.pile] == ['sku', 'cost', 'weight', 'price']
    assert [entry.name for entry in client.measures] == ['sku', 'cost', 'weight']
    assert [entry.name for entry in client.derivations] == ['price']

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
