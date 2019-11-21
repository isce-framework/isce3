# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .. import records


# declaration
class Selector(records.templater):
    """
    Metaclass that inspects a query declaration and collects the information necessary to build
    the corresponding SELECT expressions
    """


    # types
    from .Schemer import Schemer as pyre_schemer
    from .FieldReference import FieldReference as fieldReference


    # data
    pyre_reserved = {"where", "group", "order"} # SQL keywords that can't be used as field names


    # meta-methods
    @classmethod
    def __prepare__(cls, name, bases, hidden=False, **kwds):
        """
        Build an attribute table that contains the local table aliases
        """
        # chain up
        attributes = super().__prepare__(name, bases, **kwds)
        # if this is an internal class, do no more
        if hidden: return attributes

        # storage for the table aliases
        aliases = {}

        # look through the {kwds} for table aliases
        for name, value in kwds.items():
            # if {value} is a table
            if isinstance(value, cls.pyre_schemer):
                # derive a new table from it
                alias = cls.pyre_schemer(name, (value,), {})
                # save the table alias
                aliases[name] = value
                # and make it accessible as an attribute
                attributes[name] = alias

        # now, go through each of the bases to make tables from ancestor queries available in
        # the local scope of the class declaration so users don't have to stand on their head
        # to get access to them
        for base in bases:
            # skip bases that are not queries
            if not isinstance(base, cls): continue
            # queries contribute their aliased tables to my attributes
            attributes.update(base.pyre_tables)

        # prime the table aliases
        attributes["pyre_tables"] = aliases

        # return the attribute container
        return attributes


    def __new__(cls, name, bases, attributes, hidden=False, **kwds):
        # chain up
        query = super().__new__(cls, name, bases, attributes, **kwds)
        # if this is an internal class, do no more
        if hidden: return query

        # pile of tables referenced by this query
        tables = {}

        # go through the superclasses
        for base in reversed(query.__mro__):
            # narrow the search down to queries
            if not isinstance(base, cls): continue
            # collect the table references
            tables.update(base.pyre_tables)

        # record the referenced tables
        query.pyre_tables = tables

        # all done
        return query


# end of file
