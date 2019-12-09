# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
from . import gsl
# superclass
from .Vector import Vector


# declaration
class VectorView(Vector):
    """
    A view into the data of another vector
    """


    # meta-methods
    def __init__(self, vector, start, shape, **kwds):
        # adjust the parameters, just in case
        start = int(start)
        shape = int(shape)
        # store a reference to the underlying vector so it lives long enough
        self.vector = vector
        # build the view
        self.view, data = gsl.vector_view_alloc(vector.data, start, shape)
        # chain up
        super().__init__(shape=shape, data=data, **kwds)

        # all done
        return


# end of file
