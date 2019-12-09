#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that chart class records get built as expected
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
        sku = pyre.tabular.inferred(sheet.sku)

    # check the sheet class
    assert chart.pyre_sheets == {'sheet': sales}
    # check the dimensions
    assert chart.pyre_localDimensions == (chart.sku,)
    assert chart.pyre_dimensions == (chart.sku,)

    # and return the chart
    return chart


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
