#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Populate a simple sheet
"""


def test():
    import pyre.tabular

    class pricing(pyre.tabular.sheet):
        """
        The sheet layout
        """
        # layout
        sku = pyre.tabular.str()
        description = pyre.tabular.str()
        production = pyre.tabular.float()
        shipping = pyre.tabular.float()
        margin = pyre.tabular.float()
        overhead = pyre.tabular.float()


    # our data set
    data = [
        ("4000", "tomatoes", 2.95, 5, .2, 50),
        ("4001", "peppers", 0.35, 15, .1, 25),
        ("4002", "grapes", 1.65, 15, .15, 15),
        ("4003", "kiwis", 0.95, 7, .15, 75),
        ("4004", "lemons", 0.50, 4, .25, 50),
        ("4005", "oranges", 0.50, 4, .25, 50),
        ]
    # make a sheet
    p = pricing(name="vegetables")
    # and populate it
    p.pyre_immutable(data)

    # check that all is good
    for expected, actual in zip(data, p):
        assert expected == tuple(actual)

    # and return the data set
    return p


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
