# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the pyre package
import pyre
# my ancestors
from .LineMill import LineMill
from .Expression import Expression


# my declaration
class SQL(LineMill, Expression):
    """
    Support for SQL
    """


    # traits
    languageMarker = pyre.properties.str(default='SQL')
    languageMarker.doc = "the variant to use in the language marker"


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # access the {operator} module
        import operator
        # adjust the symbol table
        self._symbols[operator.pow] = "^"
        self._symbols[operator.eq] = "="
        self._symbols[operator.ne] = "<>"
        self._symbols[operator.abs] = "@"
        self._symbols[operator.and_] = "AND"
        self._symbols[operator.or_] = "OR"
        # all done
        return


    # overrides
    def _literalRenderer(self, node, **kwds):
        """
        Render {node} as a literal
        """
        # get the value of the node
        value = node._value
        # if it is already a string
        if isinstance(value, str):
            # just escape the single quotes
            return "'{}'".format(value.replace("'", "''"))

        # check whether
        try:
            # it knows how to render itself
            return value.sql()
        # otherwise
        except AttributeError:
            # no worries
            pass

        # last resort: render the value as a string
        return str(value)


    # private data
    comment='--'


# end of file
