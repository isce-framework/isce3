# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import operator


# declaration
class Ordering:
    """
    This is a mix-in class that traps comparisons

    The point is to redirect comparisons among instances of subclasses of {Ordering} to methods
    defined in these subclasses. These methods then build and return representations of the
    corresponding operators and their operands.

    {Ordering} expects its subclasses to define two class methods: {literal} and
    {operator}. The former is used to encapsulate operands that are not {Ordering}
    instances. The latter is used to construct the operator representations
    """


    # overrides for the python standard methods
    # methods are listed in the order they show up in the python documentation
    def __eq__(self, other):
        # if {other} is not a node
        if not isinstance(other, Ordering):
            # promote it
            other = self.literal(value=other)
        # build a representation of the equality test
        return self.operator(evaluator=operator.eq, operands=(self, other))


    # and of course, now that we have overridden __eq__, we must specify this so that
    # {Ordering} instances can be keys of dictionaries and members of sets...
    __hash__ = object.__hash__


    def __ne__(self, other):
        # if {other} is not a node
        if not isinstance(other, Ordering):
            # promote it
            other = self.literal(value=other)
        # build a representation of the inequality test
        return self.operator(evaluator=operator.ne, operands=(self, other))


    def __le__(self, other):
        # if {other} is not a node
        if not isinstance(other, Ordering):
            # promote it
            other = self.literal(value=other)
        # build a representation of {<=}
        return self.operator(evaluator=operator.le, operands=(self, other))


    def __ge__(self, other):
        # if {other} is not a node
        if not isinstance(other, Ordering):
            # promote it
            other = self.literal(value=other)
        # build a representation of the equality test
        return self.operator(evaluator=operator.ge, operands=(self, other))


    def __lt__(self, other):
        # if {other} is not a node
        if not isinstance(other, Ordering):
            # promote it
            other = self.literal(value=other)
        # build a representation of the equality test
        return self.operator(evaluator=operator.lt, operands=(self, other))


    def __gt__(self, other):
        # if {other} is not a node
        if not isinstance(other, Ordering):
            # promote it
            other = self.literal(value=other)
        # build a representation of the equality test
        return self.operator(evaluator=operator.gt, operands=(self, other))


# end of file
