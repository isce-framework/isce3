# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the pyre package
import pyre
# my ancestors
from .BlockMill import BlockMill
from .Expression import Expression


# my declaration
class C(BlockMill, Expression):
    """
    Support for C
    """


    # traits
    languageMarker = pyre.properties.str(default='C')
    languageMarker.doc = "the variant to use in the language marker"


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # access the {operator} module
        import operator
        # adjust the symbol table
        self._symbols[operator.floordiv] = "/"
        self._symbols[operator.and_] = "&&"
        self._symbols[operator.or_] = "||"
        # and the rendering strategy table
        self._renderers[operator.pow] = self._powerRenderer
        # all done
        return


    # implementation details
    def _powerRenderer(self, node):
        """
        Render {node.op1} raised to the {node.op2} power
        """
        # get the base and the exponent
        base, exponent = node.operands
        # render my operands
        op1 = self._renderers[type(base)](base)
        op2 = self._renderers[type(exponent)](exponent)
        # and return my string
        return "pow({},{})".format(op1, op2)


    # private data
    startBlock = '/*'
    commentMarker = ' *'
    endBlock = '*/'


# end of file
