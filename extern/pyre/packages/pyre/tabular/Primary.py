# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# super class
from .Column import Column


# declaration
class Primary(Column):
    """
    A selector that serves a column of primary keys, i.e. a column all of whose vales are
    distinct and can be used as an index into the sheet
    """


    # interface
    def refresh(self):
        """
        Rebuild my row map
        """
        # easy enough
        self.rowmap = self.prime()
        # all done
        return self


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my row map
        self.rowmap = self.prime()
        # all done
        return


    def __getitem__(self, value):
        """
        Retrieve the sheet row that contains {value} in my column
        """
        # get the row number
        row = self.rowmap[value]
        # and return the associated record
        return self.sheet.pyre_data[row]


    # implementation details
    def prime(self):
        """
        Build my value index
        """
        # initialize the index
        index = {}
        # go through all the rows in my sheet
        for row, record in enumerate(self.sheet.pyre_data):
            # get the value of my field
            value = record[self.index]
            # map the value to the row number
            index[value] = row
        # all done
        return index


# end of file
