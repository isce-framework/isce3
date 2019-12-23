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
    def pyre_make(self, **kwds):
        """
        Build all products
        """

    @pyre.provides
    def pyre_tasklist(self, **kwds):
        """
        Generate the sequence of factories that must be invoked to rebuild a product
        """


    @pyre.provides
    def pyre_targets(self, **kwds):
        """
        Generate the set of products that must be refreshed while rebuilding a product
        """


# end of file
