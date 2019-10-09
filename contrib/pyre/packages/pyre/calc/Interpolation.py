# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re # for the formula compiler
import weakref # to keep a reference to my model
import operator # for the computation of my value
import functools # for the computation of my value


# my declaration
class Interpolation:
    """
    Support for building evaluation graphs involving the values of nodes registered with a
    {SymbolTable} instance. {Interpolation} builds its value by splicing together strings.
    """


    # types
    from .exceptions import (
        CircularReferenceError,
        EmptyExpressionError, ExpressionSyntaxError, UnresolvedNodeError,
        EvaluationError )


    # constants
    category = "interpolation"
    # public data
    expression = None # the expression supplied by the client


    # classifiers
    @property
    def interpolations(self):
        """
        Return a sequence over the nodes in my dependency graph that are constructed by expanding
        the values of other nodes in a macro
        """
        # i am one
        yield self
        # nothing further
        return


    # value management
    def getValue(self):
        """
        Compute and return my value
        """
        # compute the values of my operands
        values = (str(op.value) for op in self.operands)
        # apply my operator
        return functools.reduce(operator.add, values)


    def setValue(self, value):
        """
        Use the new {value} as my formula
        """
        # adjust my state
        self.expression = value
        self._operands = tuple(self.compile(model=self._model, expression=value))
        # all done
        return self


    # support for graph traversals
    def identify(self, authority, **kwds):
        """
        Let {authority} know I am an interpolation
        """
        # invoke the callback
        return authority.onInterpolation(interpolation=self, **kwds)


    # meta-methods
    def __init__(self, model, expression, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my local state
        self.expression = expression
        self._model = weakref.proxy(model)
        # all done
        return


    # implementation details
    @classmethod
    def compile(cls, model, expression):
        """
        Compile {expression} and build an evaluator that resolves named references to other
        nodes against {model}.
        """
        # if {expression} is empty
        if not expression:
            # complain
            raise cls.EmptyExpressionError(formula=expression)

        # initialize the offset into the expression
        pos = 0
        # storage for my operands
        operands = []
        # initial portion of the expression
        fragment = ''
        # iterate over all the matches
        for match in cls._scanner.finditer(expression):
            # get the extent of the match
            start, end = match.span()
            # save the current string fragment
            fragment += expression[pos:start]
            # if this is an escaped '{'
            if match.group('esc_open'):
                # add a single '{' to the fragment
                fragment += '{'
            # if this is an escaped '}'
            elif match.group('esc_close'):
                # add a single '}' to the fragment
                fragment += '}'
            # unmatched braces
            elif match.group("lone_open") or match.group("lone_closed"):
                raise cls.ExpressionSyntaxError(
                    formula=expression,
                    error="unmatched {!r}".format(match.group()))
            # otherwise
            else:
                # it must be an identifier
                identifier = match.group('identifier')
                # if the current fragment is not empty, turn it into a variable node
                if fragment: operands.append(model.literal(value=fragment))
                # reset the fragment
                fragment = ''
                # use the identifier to locate the associated node
                reference = model.retrieve(identifier)
                # add it to my operands
                operands.append(reference)
            # update the location in {expression}
            pos = end

        # store the trailing part of the expression
        fragment += expression[pos:]

        # if there were no matches, the expression had no node evaluations; but since it may
        # have had escaped braces, make sure the caller has access to the processed value
        if not operands:
            # complain
            raise cls.EmptyExpressionError(formula=fragment)

        # and if it's not empty, turn it into a variable
        if fragment: operands.append(model.literal(value=fragment))

        # summarize
        # print(" ** SymbolTable.interpolation:")
        # print("    expression:", expression)
        # print("    operands:", operands)

        # all done
        return operands


    @classmethod
    def expand(cls, model, expression):
        """
        Compute the value of {expression} by expanding any references to {model} nodes
        """
        # compile the expression; pass any exceptions through to the caller
        operands = cls.compile(model=model, expression=expression)
        # compute the values of the operands
        values = (op.value for op in operands)
        # splice them together and return the result
        return functools.reduce(operator.add, values)


    # private data
    _scanner = re.compile( # the expression tokenizer
        r"(?P<esc_open>{{)"
        r"|"
        r"(?P<esc_close>}})"
        r"|"
        r"{(?P<identifier>[^}{]+)}"
        r"|"
        r"(?P<lone_open>{)"
        r"|"
        r"(?P<lone_closed>})"
        )

# end of file
