# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools
# support
import pyre
# protocols
from .Flow import Flow
from .Producer import Producer
from .Specification import Specification
# my superclass
from .Factory import Factory


# class declaration
class DynamicWorkflow(Factory, family='pyre.flow.dynamic', implements=Flow):
    """
    A container of flow products and factories specified at runtime
    """


    # user configurable state
    factories = pyre.properties.set(schema=Producer())
    factories.doc = "the set of my factories"

    products = pyre.properties.set(schema=Specification())
    products.doc = "the set of my products"


# end of file
