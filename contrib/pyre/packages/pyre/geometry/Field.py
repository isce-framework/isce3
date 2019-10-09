# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my base class
from .Grid import Grid


# class declaration
class Field(Grid):
    """
    A container of mesh cell values
    """


    # meta-methods
    def __init__(self, name, *args, **kwds):
        # chain up
        super().__init__(*args, **kwds)
        # save my name
        self.name = name
        # all done
        return


# end of file
