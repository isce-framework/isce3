# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Constraint:
    """
    The base class for constraints
    """


    # exceptions
    from .exceptions import ConstraintViolationError


    # interface
    def validate(self, value, **kwds):
        """
        The default behavior for constraints is to raise a ConstraintViolationError.

        Override to implement a specific test
        """
        # complain; all subclasses should chain up, and this the end of the line
        raise self.ConstraintViolationError(self, value)


    # function interface
    def __call__(self, value, **kwds):
        """
        Interface to make constraints callable
        """
        # forward to my method
        return self.validate(value=value, **kwds)


    # logical operations
    def __and__(self, other):
        """
        Enable the chaining of constraints using the logical operators
        """
        # get the operator
        from .And import And
        # build a constraint and return it
        return And(self, other)


    def __or__(self, other):
        """
        Enable the chaining of constraints using the logical operators
        """
        # get the operator
        from .Or import Or
        # build a constraint and return it
        return Or(self, other)


# end of file
