#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Configure an interval dimension
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
        quantity = pyre.tabular.interval(measure=sales.quantity, interval=(0, 20), subdivisions=4)

    # check that the instance picked up the expected dimensions
    assert chart.quantity.interval == (0, 20)
    assert chart.quantity.subdivisions == 4
    # adjust the binning strategy
    chart.quantity.subdivisions = 2
    # check that theconfiguration was updated
    assert chart.quantity.subdivisions == 2

    # and return the chart object
    return chart


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
