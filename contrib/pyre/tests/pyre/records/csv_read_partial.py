#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Load partial records from a csv file
"""


def test():
    import pyre.records

    # layout the record
    class item(pyre.records.record):
        # the fields
        sku = pyre.records.str()
        margin = pyre.records.float()
        description = pyre.records.str()

    # build the target tuple
    target = [
        ("4000", 50, "tomatoes"),
        ("4001", 25, "peppers"),
        ("4002", 15, "grapes"),
        ("4003", 75, "kiwis"),
        ("4004", 50, "lemons"),
        ("4005", 50, "oranges"),
        ]

    # create the reader
    csv = pyre.records.csv()
    # read the csv data
    source = csv.immutable(layout=item, uri="vegetables.csv")
    # check
    for given, loaded in zip(target, source):
        assert given == loaded

    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
