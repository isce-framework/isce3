# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .. import records


# declaration
class Column(records.selector):
    """
    A selector that grants access to sheet columns
    """


    # public data
    sheet = None # the worksheet I am bound to


    # meta-methods
    def __init__(self, sheet, **kwds):
        # chain up
        super().__init__(**kwds)
        # bind my to the sheet
        self.sheet = sheet
        # all done
        return


    def __iter__(self):
        """
        Build an iterator over the values in this column
        """
        # go through all the records
        for record in self.sheet.pyre_data:
            # yielding the values of my column
            yield record[self.index]
        # all done
        return


# end of file
