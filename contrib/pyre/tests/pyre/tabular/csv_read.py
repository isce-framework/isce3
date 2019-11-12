#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Read a sheet from a csv file
"""


def test():
    # get the package
    import pyre.tabular
    # lay out a sheet
    class pricing(pyre.tabular.sheet):
        """
        The sheet layout
        """
        # layout
        sku = pyre.tabular.str()
        description = pyre.tabular.str()
        production = pyre.tabular.float()
        overhead = pyre.tabular.float()
        shipping = pyre.tabular.float()
        margin = pyre.tabular.float()

    # make a csv reader
    csv = pyre.tabular.csv()
    # build the data set
    data = csv.read(layout=pricing, uri='vegetables.csv')
    # make a sheet instance
    sheet = pricing(name="vegetables")
    # populate the table
    sheet.pyre_immutable(data=data)
    # check that we read the data correctly
    # here is what we expect
    target = [
        ("4000", "tomatoes", 2.95, 5, .2, 50),
        ("4001", "peppers", 0.35, 15, .1, 25),
        ("4002", "grapes", 1.65, 15, .15, 15),
        ("4003", "kiwis", 0.95, 7, .15, 75),
        ("4004", "lemons", 0.50, 4, .25, 50),
        ("4005", "oranges", 0.50, 4, .25, 50),
        ]
    # compare with what we extracted
    for expected, loaded in zip(target, sheet):
        assert expected == tuple(loaded)
    # and return the sheet
    return sheet


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
