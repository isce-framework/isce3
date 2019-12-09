# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Element import Element


# class declaration
class ElementContainer:
    """
    The base class for HTML elements that can contain other elements
    """


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # make storage for the contents
        self.contents = []
        # all done
        return


# end of file
