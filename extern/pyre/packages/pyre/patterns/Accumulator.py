# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# base class
from .CoFunctor import CoFunctor


# class declaration
class Accumulator(CoFunctor):
    """
    A coroutine that accumulates data in a container
    """


    # interface
    def throw(self, errorTp, error=None, traceback=None):
        """
        Handle exceptions
        """
        # accumulators ignore errors
        return


    # meta-methods
    def __init__(self, **kwds):
        # initialize my cache
        self.cache = []
        # chain up
        super().__init__(**kwds)
        # all done
        return


    # my coroutine
    def __call__(self):
        """
        Store everything that comes in
        """
        # for ever
        while True:
            # get the item
            item = yield
            # store it
            self.cache.append(item)
        # all done
        return


# end of file
