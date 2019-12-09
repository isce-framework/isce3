#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Build a couple of views
"""


def test():
    # get the package
    import pyre.tabular

    # lay out a sheet
    class cost(pyre.tabular.sheet):
        """The prices of things"""
        # layout
        sku = pyre.tabular.str()
        sku.index = True

        description = pyre.tabular.str()
        production = pyre.tabular.float()
        overhead = pyre.tabular.float()
        shipping = pyre.tabular.float()
        margin = pyre.tabular.float()

    # and another
    class sales(pyre.tabular.sheet):
        """The transaction data"""
        # layout
        date = pyre.tabular.str()
        time = pyre.tabular.str()
        sku = pyre.tabular.str()
        quantity = pyre.tabular.float()
        discount = pyre.tabular.float()
        sale = pyre.tabular.float()


    # make a csv reader
    csv = pyre.tabular.csv()
    # make a couple of sheets
    vegetables = cost(name="vegetables")
    transactions = sales(name="sales")

    # populate them
    csv.immutable(layout=transactions, uri="sales.csv")
    csv.immutable(layout=vegetables, uri="vegetables.csv")

    # and return the sheets
    return vegetables, sales


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
