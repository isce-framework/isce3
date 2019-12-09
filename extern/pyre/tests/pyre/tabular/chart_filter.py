#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise an inferred chart dimension
"""


def test():
    # get the package
    import pyre.tabular

    # make a sheet
    class sales(pyre.tabular.sheet):
        """The transaction data"""
        # layout
        date = pyre.tabular.str()
        time = pyre.tabular.str()
        sku = pyre.tabular.str()
        quantity = pyre.tabular.float()
        discount = pyre.tabular.float()
        sale = pyre.tabular.float()

    # make a chart
    class chart(pyre.tabular.chart, sheet=sales):
        """
        Aggregate the information in the {sales} table
        """
        sku = pyre.tabular.inferred(sales.sku)
        date = pyre.tabular.inferred(sales.date)


    # make a csv reader
    csv = pyre.tabular.csv()
    # build a dataset
    data = csv.read(layout=sales, uri='sales.csv')
    # make a sheet
    transactions = sales(name="sales").pyre_immutable(data)

    # build a chart
    cube = chart(sheet=transactions)

    # select the records the match a given sku and date
    grp = cube.pyre_filter(date="2010/11/01", sku="4000")
    # check that we got what we expected
    assert grp == {0, 5, 6}

    # and return the chart and the sheet
    return cube, transactions


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
