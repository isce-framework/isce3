# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .. import records


class Selector(records.selector):
    """
    The basic selector that provides named access to sheet columns
    """


    # types
    from .Column import Column as column # access to the entries of a given column
    from .Primary import Primary as primary # access to an indexed column


    # meta-methods
    def __get__(self, sheet, cls):
        """
        Read access to the field slice from the data set
        """
        # if this is access as a class attribute, return the field descriptor
        if sheet is None: return self.field

        # otherwise, this is access to a sheet instance
        # if I manage a primary field
        if self.field._primary:
            # make an indexed column and return it
            return self.primary(sheet=sheet, field=self.field, index=self.index)
        # otherwise, bind a regular column selector to this instance and return it
        return self.column(sheet=sheet, field=self.field, index=self.index)


# end of file
