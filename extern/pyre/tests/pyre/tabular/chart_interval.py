#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise chart interval dimensions
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

    # and a chart
    class chart(pyre.tabular.chart, sheet=sales):
        """
        Aggregate the information in the {sales} table
        """
        quantity = pyre.tabular.interval(measure=sales.quantity, interval=(0, 20), subdivisions=4)


    # make a csv reader
    csv = pyre.tabular.csv()
    # build a dataset
    data = csv.read(layout=sales, uri='sales.csv')
    # make a sheet
    transactions = sales(name="sales").pyre_immutable(data)

    # build a chart
    cube = chart(sheet=transactions)
    # bin the transactions
    quantities = cube.quantity

    # there should have been no rejects in this sample dataset
    assert quantities.rejects == []
    # check that all transactions were binned
    assert len(transactions) == sum(len(bin) for bin in quantities)

    # and return the chart object
    return cube


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
