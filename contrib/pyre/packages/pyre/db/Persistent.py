# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import weakref
# superclass
from ..patterns.AttributeClassifier import AttributeClassifier


# class declaration
class Persistent(AttributeClassifier):
    """
    Metaclass that enables the creation of classes whose instances store part of their
    attributes in relational database tables.

    {Persistent} and its instance {Object} provide the necessary layer to bridge object
    oriented semantics with the relational model. The goal is to make the existence of the
    relational tables more transparent to the developer of database applications by removing as
    much of the grunt work of storing and retrieving object state as possible.
    """


    # meta-methods
    def __init__(self, name, bases, attributes, schema=None, **kwds):
        # chain up
        super().__init__(name, bases, attributes, **kwds)

        # if i model a table
        if schema is not None:
            # attach my schema
            self.pyre_primaryTable = schema
            # register me with the schema manager
            self.pyre_schema.models[schema] = self
            # initialize my extent
            self.pyre_extent = weakref.WeakValueDictionary()

        # all done
        return


    def __call__(self, **kwds):
        """
        Create one of my instances
        """
        # show me
        # print('pyre.db.Persistent.__call__: kwds={}'.format(kwds))
        # chain up to build the instance
        model = super().__call__()
        # and return it
        return model


# end of file
