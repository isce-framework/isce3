# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# base class
from .CoFunctor import CoFunctor


# class declaration
class Printer(CoFunctor):
    """
    A coroutine that sends a textual representation of the values it receives to {stdout}
    """


    # interface
    def throw(self, errorTp, error=None, traceback=None):
        """
        Handle exceptions
        """
        # printers ignore errors
        return


    # meta-methods
    def __init__(self, format=None, **kwds):
        # save the format
        self.format = format
        # chain up
        super().__init__(**kwds)
        # all done
        return


    # my coroutine
    def __call__(self):
        """
        Store everything that comes in
        """
        # get my format
        format = self.format
        # for ever
        while True:
            # get the item
            item = yield
            # format it
            output = item if format is None else format.format(item)
            # store it
            print(output)
        # all done
        return


# end of file
