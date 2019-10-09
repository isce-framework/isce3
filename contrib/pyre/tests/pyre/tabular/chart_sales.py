#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Read in a couple of tables and build a rudimentary chart
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

    # build a chart
    class chart(pyre.tabular.chart, sheet=sales):
        """
        Aggregate the information in the {sales} table
        """
        sku = pyre.tabular.inferred(sales.sku)


    # make a csv reader
    csv = pyre.tabular.csv()
    # build a data set
    data = csv.read(layout=sales, uri='sales.csv')
    # make a sheet
    transactions = sales(name="sales").pyre_immutable(data)

    # build a chart
    cube = chart(sheet=transactions)
    # bin the skus
    skus = cube.sku

    # here are the skus we expect to retrieve from the data set
    targets = {"4000", "4001", "4002", "4003", "4004", "4005"}
    # check that the skus were classified correctly
    assert set(skus.keys()) == targets
    # check that all the transactions were binned
    assert len(transactions) == sum(len(bin) for bin in skus.values())

    # verify that all transaction records binned as having a given sku do so
    for sku, bin in skus.items():
        for rank in bin:
            assert transactions[rank].sku == sku

    # and return the charts and the sheets
    return cube, transactions


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
