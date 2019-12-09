# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# base class
from .CoFunctor import CoFunctor


# class declaration
class Tee(CoFunctor, list):
    """
    A coroutine that accumulates data in a container
    """


    # generator interface
    def throw(self, errorTp, error=None, traceback=None):
        """
        Raise an exception
        """
        # pass it along
        for sink in self: sink.throw(errorTp, error, traceback)
        # all done
        return


    def close(self):
        """
        Shut things down
        """
        # pass it along
        for sink in self: sink.close()
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
            # propagate
            for sink in self: sink.send(item)
        # all done
        return


# end of file
