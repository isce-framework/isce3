#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise column extraction
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
    # populate it
    p.pyre_immutable(data)

    # check the fields in the sheet against the data set
    assert tuple(p.sku) == tuple(record[0] for record in data)
    assert tuple(p.description) == tuple(record[1] for record in data)
    assert tuple(p.production) == tuple(record[2] for record in data)
    assert tuple(p.shipping) == tuple(record[3] for record in data)
    assert tuple(p.margin) == tuple(record[4] for record in data)
    assert tuple(p.overhead) == tuple(record[5] for record in data)

    # compute the average production cost and check we got it right
    assert pyre.patterns.mean(p.production) == sum(entry[2] for entry in data)/len(data)

    # and return the data set
    return p


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
