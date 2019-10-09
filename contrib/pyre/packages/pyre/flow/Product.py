# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre
# my protocol
from .Specification import Specification
# my superclass
from .Node import Node


# class declaration
class Product(Node, implements=Specification):
    """
    The base class for data products
    """


    # interface
    def sync(self):
        """
        Examine my state
        """


# end of file
