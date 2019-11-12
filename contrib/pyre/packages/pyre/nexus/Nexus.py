# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
# base class
from .Asynchronous import Asynchronous


# declaration
class Nexus(Asynchronous, family="pyre.nexus.servers"):
    """
    Protocol definition for components that enable applications to interact over the network
    """


    # default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        """
        The default implementation of the {Nexus} protocol
        """
        # get my favorite
        from .Node import Node
        # return it
        return Node


# end of file
