# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre
# protocols
from .Producer import Producer
from .Specification import Specification


# class declaration
class Workflow(pyre.application, family='pyre.applications.workflow'):
    """
    A simple application class for managing workflows
    """


    # user configurable state
    factories = pyre.properties.set(schema=Producer())
    factories.doc = "the set of flow factories"

    products = pyre.properties.set(schema=Specification())
    products.doc = "the set of flow products"


# end of file
