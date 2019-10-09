# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Dimension import Dimension


# declaration
class Interval(Dimension):
    """
    A chart axis whose tick marks are intervals of (a subset) of the range of the values in a
    given sheet column.
    """


    # meta-methods
    def __get__(self, chart, cls):
        # if I am being accessed through an instance
        if chart:
            # bin my measure over the chart and return it
            return self.axis(chart=chart, dimension=self)
        # otherwise, just return myself
        return self


    # meta-methods
    def __init__(self, interval, subdivisions, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my layout
        self.interval = interval
        self.subdivisions = subdivisions
        # all done
        return


    # implementation details
    class axis:

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

            # the geometry of my bins
            start, end = dimension.interval
            subdivisions = dimension.subdivisions
            width = (end - start) / subdivisions

            # build my bins
            self.bins = tuple(set() for bin in range(subdivisions))
            # and a list of records that were rejected because they are outside my interval
            self.rejects = []

            # get the records
            for row, record in enumerate(sheet.pyre_data):
                # get the value of my measure
                value = record[column]
                # bin it
                rank = int((value - start)/width)
                # check whether it falls within my bounds
                if 0 <= rank < subdivisions:
                    # place it in its bin
                    self.bins[rank].add(row)
                # otherwise
                else:
                    # reject it
                    self.rejects.append(row)

            # all done
            return

        def __len__(self):
            # easy enough
            return len(self.bins)

        def __iter__(self):
            # also easy
            return iter(self.bins)

        def __getitem__(self, bin):
            # also easy
            return self.bins[bin]


# end of file
