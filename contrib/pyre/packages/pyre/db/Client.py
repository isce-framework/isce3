# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre
# and my protocols
from .DataStore import DataStore


# declaration
class Client(pyre.component, family="pyre.db.client"):
    """
    The base class for components that connect to data stores
    """

    # user configurable state
    server = DataStore()


# end of file
