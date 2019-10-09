# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# metaclass
from .Schemer import Schemer
# superclass
from .. import records


# declaration
class Table(records.record, metaclass=Schemer):
    """
    Base class for database table declarations
    """


    # constants
    from .literals import default, null
    from .expressions import IsNull as isNull, IsNotNull as isNotNull


    # interface
    # declaration decorators
    @classmethod
    def pyre_primaryKey(cls, reference):
        """
        Add {reference} to the tuple of fields that must be marked as primary keys
        """
        # add it to the pile
        cls._pyre_primaryKeys.add(reference.field)
        # and return
        return cls


    @classmethod
    def pyre_unique(cls, reference):
        """
        Add {reference} to the tuple of fields that must be marked as unique
        """
        # add it to the pile
        cls._pyre_uniqueFields.add(reference.field)
        # and return
        return cls


    @classmethod
    def pyre_foreignKey(cls, field, foreign):
        """
        Mark {field} as a reference to {foreign}
        """
        # add an entry to the foreign key list
        cls._pyre_foreignKeys.append( (field, foreign) )
        # and return
        return cls


    @classmethod
    def pyre_check(cls, expression):
        """
        Add {expression} to the list of my nameless constraints
        """
        # add {expression} to my pile of constraints
        cls._pyre_constraints.append(expression)
        # and return
        return cls


    # implementation details
    # table attributes that generate the table wide declaration statements; initialized by my
    # metaclass at compile time
    _pyre_primaryKeys = None
    _pyre_uniqueFields = None
    _pyre_foreignKeys = None
    _pyre_constraints = None


# end of file
