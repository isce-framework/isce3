#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Load records from a csv file
"""


def test():
    import pyre.records

    # layout the record
    class item(pyre.records.record):
        # the fields
        sku = pyre.records.str()
        description = pyre.records.str()
        production = pyre.records.float()
        overhead = pyre.records.float()
        shipping = pyre.records.float()
        margin = pyre.records.float()
        # a derived quantity
        price = production*(1 + overhead/100 + margin/100) + shipping

    # build the target tuple
    target = [
        ("4000", "tomatoes", 2.95, 5, .2, 50, 2.95*(1+.05+.5)+.2),
        ("4001", "peppers", 0.35, 15, .1, 25, .35*(1+.15+.25)+.1),
        ("4002", "grapes", 1.65, 15, .15, 15, 1.65*(1+.15+.15)+.15),
        ("4003", "kiwis", 0.95, 7, .15, 75, .95*(1+.07+.75)+.15),
        ("4004", "lemons", 0.50, 4, .25, 50, .5*(1+.04+.5)+.25),
        ("4005", "oranges", 0.50, 4, .25, 50, .5*(1+.04+.5)+.25),
        ]

    # create the reader
    csv = pyre.records.csv()
    # read the csv data
    source = csv.mutable(layout=item, uri="vegetables.csv")
    # check
    for given, loaded in zip(target, source):
        # check access via both names and indices
        assert loaded.sku == loaded[0]
        assert loaded.description == loaded[1]
        assert loaded.production == loaded[2]
        assert loaded.overhead == loaded[3]
        assert loaded.shipping == loaded[4]
        assert loaded.margin == loaded[5]
        # and compare with the target data
        assert given == tuple(loaded)

    return item


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
