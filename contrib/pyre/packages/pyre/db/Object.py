# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orhologue
# (c) 1998-2019 all rights reserved
#


# my superclass
from ..framework.Dashboard import Dashboard
# my metaclass
from .Persistent import Persistent


# class declaration
class Object(Dashboard, metaclass=Persistent):
    """
    The base class of classes whose instances store part of their attributes in relational
    database tables.

    {Object} and its class {Persistent} provide the necessary layer to bridge object oriented
    semantics with the relational model. The goal is to make the existence of the relational
    tables more transparent to the developer of database applications by removing as much of
    the grunt work of storing and retrieving object state as possible.
    """


    # implementation details
    pyre_extent = None # a map from primary keys to model instances
    pyre_primaryTable = None # the table I model


# end of file
