# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Definitions for all the exceptions raised by this package
"""


# pull the exceptions from {algebraic}
from ..algebraic.exceptions import NodeError, CircularReferenceError

# the local ones
class EvaluationError(NodeError):
    """
    Base class for node evaluation exceptions
    """

    # public data
    description = "evaluation error: {0.error}"

    # meta-methods
    def __init__(self, error, node=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.node = node
        self.error = error
        # all done
        return


class ExpressionError(NodeError):
    """
    Base class for expression errors; useful when trapping them as a category
    """


class EmptyExpressionError(ExpressionError):
    """
    Exception raised when the expression factory did not encounter any named references to
    other nodes
    """

    # public data
    description = "while parsing {0.expression!r}: no references found"

    # meta-methods
    def __init__(self, formula, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.expression = formula
        # all done
        return


class ExpressionSyntaxError(ExpressionError):
    """
    Exception raised when the python interpreter encounters a syntax error while compiling the
    expression
    """

    # public data
    description = "while evaluating {0.expression!r}: {0.error}"

    # meta-methods
    def __init__(self, formula, error, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.expression = formula
        self.error = error
        # all done
        return


class UnresolvedNodeError(NodeError):
    """
    Signal a value request from an unresolved node
    """

    # public data
    description = "node {0.name!r} is unresolved"

    # meta-methods
    def __init__(self, name, node=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.name = name
        self.node = node
        # all done
        return


class AliasingError(NodeError):
    """
    Signal that an alias was requested among names that were associated with existing nodes
    """

    # public data
    description = "both {0.target!r} and {0.alias!r} have existing nodes"

    # meta-methods
    def __init__(self, key, target, alias, targetNode, targetInfo, aliasNode, aliasInfo, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.key = key
        self.target = target
        self.alias = alias
        self.targetNode = targetNode
        self.targetInfo = targetInfo
        self.aliasNode = aliasNode
        self.aliasInfo = aliasInfo
        # all done
        return


# end of file
