# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Dimension import Dimension


# declaration
class Inferred(Dimension):
    """
    A chart axis whose tick marks are the unique values found in a given sheet column
    """


    # meta-methods
    def __get__(self, chart, cls):
        # if I am being accessed through an instance
        if chart:
            # bin my measure over the chart and return it
            return self.axis(chart=chart, dimension=self)
        # otherwise, just return myself
        return self


    # implementation details
    class axis(dict):

        # meta-methods
        def __init__(self, chart, dimension, **kwds):
            # chain up
            super().__init__(**kwds)

            # get the sheet
            sheet = chart.sheet
            # the measure
            measure = dimension.measure
            # and my column number
            column = sheet.pyre_columns[measure]

            # get the records
            for row, record in enumerate(sheet.pyre_data):
                # get the value of my column
                value = record[column]
                # get the set of rows that have the same value
                rows = self.setdefault(value, set())
                # add this one to the set
                rows.add(row)

            # all done
            return


# end of file
