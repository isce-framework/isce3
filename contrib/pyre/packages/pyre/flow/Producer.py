# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre

# class declaration
class Producer(pyre.protocol):
    """
    The requirements that all factories must implement
    """

    # required interface
    @pyre.provides
    def make(self, **kwds):
        """
        Build all products
        """

    @pyre.provides
    def plan(self, **kwds):
        """
        Describe what needs to get to done to make the products
        """


# end of file
