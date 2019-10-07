#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the chart metaclass handles inheritance properly
"""


def test():
    import pyre.tabular

    class sales(pyre.tabular.sheet):
        """The transaction data"""
        # layout
        date = pyre.tabular.str()
        time = pyre.tabular.str()
        sku = pyre.tabular.str()
        quantity = pyre.tabular.float()
        discount = pyre.tabular.float()
        sale = pyre.tabular.float()

    class base(pyre.tabular.chart, sheet=sales):
        """Aggregate the {sku} information in the {sales} table """
        sku = pyre.tabular.inferred(sales.sku)

    class chart(base):
        """Further aggregation by {date}"""
        date = pyre.tabular.inferred(base.sheet.date)

    # base checks
    # check the sheet class
    assert base.pyre_sheets == {'sheet': sales}
    # check the dimensions
    assert base.pyre_localDimensions == (chart.sku,)
    assert base.pyre_dimensions == (chart.sku,)

    # derived class checks
    # check the sheet class
    assert chart.pyre_sheets == {'sheet': sales}
    # check the dimensions in the derived chart
    assert chart.pyre_localDimensions == (chart.date,)
    assert chart.pyre_dimensions == (chart.sku, chart.date)
    # and return the chart
    return chart


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
